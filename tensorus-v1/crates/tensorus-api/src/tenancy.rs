//! Multi-tenancy: tenants, scoped API keys with RBAC roles, per-tenant usage
//! quotas, and a persisted [`TenantRegistry`] control plane.
//!
//! ## Model
//!
//! - A **tenant** is the isolation boundary. Each tenant owns a set of datasets,
//!   stored under a composite key `{tenant}.{dataset}` so two tenants never
//!   collide and one tenant can never read another's data.
//! - **API keys** are issued per tenant with a [`Role`] (`admin`/`read_write`/
//!   `read_only`). Keys are stored **hashed** (SHA-256); the plaintext is shown
//!   once at creation.
//! - A bootstrap **system** key (from the environment) runs the control plane
//!   (create tenants, issue keys, snapshot) but cannot touch tenant data.
//! - **Quotas** cap a tenant's dataset and tensor counts.
//!
//! When no registry is configured the server runs in legacy single-key mode
//! (one key, full access, no dataset prefixing) — see [`Principal::unscoped`].

use parking_lot::Mutex;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

/// A role granted to an API key, in increasing privilege.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    ReadOnly,
    ReadWrite,
    Admin,
}

impl Role {
    /// Parse a role name (tolerant of spellings).
    pub fn parse(s: &str) -> Option<Role> {
        match s.trim().to_lowercase().replace('-', "_").as_str() {
            "admin" => Some(Role::Admin),
            "read_write" | "readwrite" | "write" | "rw" => Some(Role::ReadWrite),
            "read_only" | "readonly" | "read" | "ro" => Some(Role::ReadOnly),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Role::Admin => "admin",
            Role::ReadWrite => "read_write",
            Role::ReadOnly => "read_only",
        }
    }

    fn can_write(&self) -> bool {
        matches!(self, Role::Admin | Role::ReadWrite)
    }
}

/// The authorization scope a request operates in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Scope {
    /// Legacy single-key mode: full access, datasets are not prefixed.
    Unscoped,
    /// Multi-tenant bootstrap admin: control plane only, no tenant data access.
    System,
    /// Data access scoped to a single tenant.
    Tenant(String),
}

/// An authenticated caller.
#[derive(Debug, Clone)]
pub struct Principal {
    pub scope: Scope,
    pub role: Role,
}

impl Principal {
    /// Legacy single-key principal (full access, no tenant scoping).
    pub fn unscoped() -> Self {
        Principal {
            scope: Scope::Unscoped,
            role: Role::Admin,
        }
    }

    /// System (control-plane) principal.
    pub fn system() -> Self {
        Principal {
            scope: Scope::System,
            role: Role::Admin,
        }
    }

    /// Tenant-scoped principal.
    pub fn tenant(tenant: impl Into<String>, role: Role) -> Self {
        Principal {
            scope: Scope::Tenant(tenant.into()),
            role,
        }
    }

    /// Whether the principal may perform write operations.
    pub fn can_write(&self) -> bool {
        self.role.can_write()
    }

    /// Whether the principal has the `Admin` role (within its scope).
    pub fn is_admin(&self) -> bool {
        matches!(self.role, Role::Admin)
    }

    /// Whether this is the control-plane (system or legacy) principal.
    pub fn is_system(&self) -> bool {
        matches!(self.scope, Scope::System | Scope::Unscoped)
    }
}

/// Per-tenant resource limits (`0` = unlimited).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Quota {
    #[serde(default)]
    pub max_datasets: usize,
    #[serde(default)]
    pub max_tensors: usize,
}

/// A tenant record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantInfo {
    pub id: String,
    #[serde(default)]
    pub quota: Quota,
    pub created_at: u64,
}

/// Stored API-key record (the key itself is only kept as a hash).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct KeyRecord {
    id: String,
    key_hash: String,
    tenant: String,
    role: Role,
    name: String,
    created_at: u64,
}

/// Public, non-secret view of an API key.
#[derive(Debug, Clone, Serialize)]
pub struct KeyMeta {
    pub id: String,
    pub tenant: String,
    pub role: Role,
    pub name: String,
    pub created_at: u64,
}

/// Quota-violation outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuotaError {
    /// Dataset limit reached (the limit value).
    Datasets(usize),
    /// Tensor limit reached (the limit value).
    Tensors(usize),
}

#[derive(Default, Clone, Copy)]
struct Usage {
    datasets: usize,
    tensors: usize,
}

#[derive(Default, Serialize, Deserialize)]
struct RegState {
    tenants: HashMap<String, TenantInfo>,
    /// Keyed by SHA-256 hash of the API key.
    keys: HashMap<String, KeyRecord>,
}

/// The persisted control plane: tenants, hashed keys, and live usage counters.
pub struct TenantRegistry {
    path: Option<PathBuf>,
    state: Mutex<RegState>,
    usage: Mutex<HashMap<String, Usage>>,
}

/// Whether a string is a valid tenant/dataset slug: 1–63 chars, lowercase
/// alphanumeric or hyphen, starting alphanumeric (Pinecone-style, and safe as a
/// path/composite component since it excludes `.` and `/`).
pub fn valid_slug(s: &str) -> bool {
    if s.is_empty() || s.len() > 63 {
        return false;
    }
    let bytes = s.as_bytes();
    if !bytes[0].is_ascii_lowercase() && !bytes[0].is_ascii_digit() {
        return false;
    }
    s.chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn hash_key(key: &str) -> String {
    let mut h = Sha256::new();
    h.update(key.as_bytes());
    to_hex(&h.finalize())
}

impl TenantRegistry {
    /// Create an in-memory registry (no persistence); used by tests.
    pub fn in_memory() -> Self {
        TenantRegistry {
            path: None,
            state: Mutex::new(RegState::default()),
            usage: Mutex::new(HashMap::new()),
        }
    }

    /// Load a registry from `path` (or start empty if absent), persisting future
    /// mutations back to it.
    pub fn load(path: PathBuf) -> Self {
        let state = std::fs::read(&path)
            .ok()
            .and_then(|b| serde_json::from_slice::<RegState>(&b).ok())
            .unwrap_or_default();
        TenantRegistry {
            path: Some(path),
            state: Mutex::new(state),
            usage: Mutex::new(HashMap::new()),
        }
    }

    fn persist(&self, state: &RegState) {
        if let Some(path) = &self.path {
            if let Ok(bytes) = serde_json::to_vec_pretty(state) {
                if let Some(parent) = path.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                let _ = std::fs::write(path, bytes);
            }
        }
    }

    /// Resolve an API key to a tenant principal, if it is a valid tenant key.
    pub fn resolve(&self, key: &str) -> Option<Principal> {
        let hash = hash_key(key);
        let state = self.state.lock();
        state
            .keys
            .get(&hash)
            .map(|rec| Principal::tenant(rec.tenant.clone(), rec.role))
    }

    /// Create a tenant. Errors if the id is invalid or already exists.
    pub fn create_tenant(&self, id: &str, quota: Quota) -> Result<TenantInfo, String> {
        if !valid_slug(id) {
            return Err(format!(
                "invalid tenant id '{id}' (use 1-63 lowercase alphanumeric or hyphen, starting alphanumeric)"
            ));
        }
        let mut state = self.state.lock();
        if state.tenants.contains_key(id) {
            return Err(format!("tenant '{id}' already exists"));
        }
        let info = TenantInfo {
            id: id.to_string(),
            quota,
            created_at: now_unix(),
        };
        state.tenants.insert(id.to_string(), info.clone());
        self.persist(&state);
        Ok(info)
    }

    /// Issue a new API key for a tenant, returning `(key_id, plaintext_key)`.
    /// The plaintext is not stored and cannot be retrieved again.
    pub fn issue_key(
        &self,
        tenant: &str,
        role: Role,
        name: &str,
    ) -> Result<(String, String), String> {
        let mut state = self.state.lock();
        if !state.tenants.contains_key(tenant) {
            return Err(format!("unknown tenant '{tenant}'"));
        }
        let mut raw = [0u8; 24];
        rand::thread_rng().fill_bytes(&mut raw);
        let plaintext = format!("tns_{}_{}", tenant, to_hex(&raw));
        let id = format!("key_{}", to_hex(&raw[..6]));
        let rec = KeyRecord {
            id: id.clone(),
            key_hash: hash_key(&plaintext),
            tenant: tenant.to_string(),
            role,
            name: name.to_string(),
            created_at: now_unix(),
        };
        state.keys.insert(rec.key_hash.clone(), rec);
        self.persist(&state);
        Ok((id, plaintext))
    }

    /// Revoke a key by id. Returns the owning tenant on success.
    pub fn revoke_key(&self, id: &str) -> Result<String, String> {
        let mut state = self.state.lock();
        let hash = state
            .keys
            .iter()
            .find(|(_, r)| r.id == id)
            .map(|(h, _)| h.clone());
        match hash {
            Some(h) => {
                let rec = state.keys.remove(&h).expect("present");
                self.persist(&state);
                Ok(rec.tenant)
            }
            None => Err(format!("unknown key id '{id}'")),
        }
    }

    /// The tenant that owns a key id, if any (for authorization checks).
    pub fn key_tenant(&self, id: &str) -> Option<String> {
        self.state
            .lock()
            .keys
            .values()
            .find(|r| r.id == id)
            .map(|r| r.tenant.clone())
    }

    pub fn tenant_exists(&self, tenant: &str) -> bool {
        self.state.lock().tenants.contains_key(tenant)
    }

    pub fn quota(&self, tenant: &str) -> Option<Quota> {
        self.state.lock().tenants.get(tenant).map(|t| t.quota)
    }

    pub fn list_tenants(&self) -> Vec<TenantInfo> {
        let mut v: Vec<TenantInfo> = self.state.lock().tenants.values().cloned().collect();
        v.sort_by(|a, b| a.id.cmp(&b.id));
        v
    }

    pub fn list_keys(&self, tenant: &str) -> Vec<KeyMeta> {
        let mut v: Vec<KeyMeta> = self
            .state
            .lock()
            .keys
            .values()
            .filter(|r| r.tenant == tenant)
            .map(|r| KeyMeta {
                id: r.id.clone(),
                tenant: r.tenant.clone(),
                role: r.role,
                name: r.name.clone(),
                created_at: r.created_at,
            })
            .collect();
        v.sort_by_key(|a| a.created_at);
        v
    }

    // --- usage / quota ---

    /// Seed a tenant's usage counters (called once at startup after recovery).
    pub fn set_usage(&self, tenant: &str, datasets: usize, tensors: usize) {
        self.usage
            .lock()
            .insert(tenant.to_string(), Usage { datasets, tensors });
    }

    /// Current `(datasets, tensors)` usage for a tenant.
    pub fn usage(&self, tenant: &str) -> (usize, usize) {
        let u = self.usage.lock();
        let v = u.get(tenant).copied().unwrap_or_default();
        (v.datasets, v.tensors)
    }

    /// Reserve a new dataset against the tenant's quota.
    pub fn try_add_dataset(&self, tenant: &str) -> Result<(), QuotaError> {
        let max = self.quota(tenant).map(|q| q.max_datasets).unwrap_or(0);
        let mut u = self.usage.lock();
        let e = u.entry(tenant.to_string()).or_default();
        if max != 0 && e.datasets >= max {
            return Err(QuotaError::Datasets(max));
        }
        e.datasets += 1;
        Ok(())
    }

    /// Reserve a new tensor against the tenant's quota.
    pub fn try_add_tensor(&self, tenant: &str) -> Result<(), QuotaError> {
        let max = self.quota(tenant).map(|q| q.max_tensors).unwrap_or(0);
        let mut u = self.usage.lock();
        let e = u.entry(tenant.to_string()).or_default();
        if max != 0 && e.tensors >= max {
            return Err(QuotaError::Tensors(max));
        }
        e.tensors += 1;
        Ok(())
    }

    pub fn remove_tensor(&self, tenant: &str) {
        let mut u = self.usage.lock();
        if let Some(e) = u.get_mut(tenant) {
            e.tensors = e.tensors.saturating_sub(1);
        }
    }

    pub fn remove_dataset(&self, tenant: &str) {
        let mut u = self.usage.lock();
        if let Some(e) = u.get_mut(tenant) {
            e.datasets = e.datasets.saturating_sub(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slug_validation() {
        assert!(valid_slug("acme"));
        assert!(valid_slug("acme-corp-1"));
        assert!(valid_slug("0a"));
        assert!(!valid_slug("Acme")); // uppercase
        assert!(!valid_slug("a.b")); // dot
        assert!(!valid_slug("a/b")); // slash
        assert!(!valid_slug("-lead")); // leading hyphen
        assert!(!valid_slug(""));
        assert!(!valid_slug(&"x".repeat(64)));
    }

    #[test]
    fn role_parse_and_order() {
        assert_eq!(Role::parse("admin"), Some(Role::Admin));
        assert_eq!(Role::parse("read-write"), Some(Role::ReadWrite));
        assert_eq!(Role::parse("RO"), Some(Role::ReadOnly));
        assert_eq!(Role::parse("nope"), None);
        assert!(Role::Admin.can_write() && Role::ReadWrite.can_write());
        assert!(!Role::ReadOnly.can_write());
    }

    #[test]
    fn create_tenant_issue_resolve_revoke() {
        let reg = TenantRegistry::in_memory();
        assert!(reg.create_tenant("acme", Quota::default()).is_ok());
        assert!(reg.create_tenant("acme", Quota::default()).is_err()); // duplicate
        assert!(reg.create_tenant("BAD", Quota::default()).is_err()); // invalid slug

        let (id, key) = reg.issue_key("acme", Role::ReadWrite, "ci").unwrap();
        let p = reg.resolve(&key).expect("resolves");
        assert_eq!(p.scope, Scope::Tenant("acme".into()));
        assert_eq!(p.role, Role::ReadWrite);
        assert!(reg.resolve("tns_acme_deadbeef").is_none()); // unknown key

        assert_eq!(reg.list_keys("acme").len(), 1);
        assert_eq!(reg.revoke_key(&id).unwrap(), "acme");
        assert!(reg.resolve(&key).is_none()); // revoked
        assert_eq!(reg.list_keys("acme").len(), 0);
    }

    #[test]
    fn issue_key_unknown_tenant_fails() {
        let reg = TenantRegistry::in_memory();
        assert!(reg.issue_key("ghost", Role::ReadOnly, "x").is_err());
    }

    #[test]
    fn quota_enforcement() {
        let reg = TenantRegistry::in_memory();
        reg.create_tenant(
            "t",
            Quota {
                max_datasets: 1,
                max_tensors: 2,
            },
        )
        .unwrap();
        assert!(reg.try_add_dataset("t").is_ok());
        assert_eq!(reg.try_add_dataset("t"), Err(QuotaError::Datasets(1)));
        assert!(reg.try_add_tensor("t").is_ok());
        assert!(reg.try_add_tensor("t").is_ok());
        assert_eq!(reg.try_add_tensor("t"), Err(QuotaError::Tensors(2)));
        reg.remove_tensor("t");
        assert!(reg.try_add_tensor("t").is_ok()); // freed one slot
        assert_eq!(reg.usage("t"), (1, 2));
    }

    #[test]
    fn unlimited_quota_when_zero() {
        let reg = TenantRegistry::in_memory();
        reg.create_tenant("t", Quota::default()).unwrap();
        for _ in 0..1000 {
            assert!(reg.try_add_tensor("t").is_ok());
        }
    }

    #[test]
    fn persistence_roundtrip() {
        let tmp = std::env::temp_dir().join(format!("tns_reg_{}.json", now_unix()));
        let key;
        {
            let reg = TenantRegistry::load(tmp.clone());
            reg.create_tenant("acme", Quota::default()).unwrap();
            let (_id, k) = reg.issue_key("acme", Role::Admin, "root").unwrap();
            key = k;
        }
        // Reload from disk; the tenant and key survive.
        let reg2 = TenantRegistry::load(tmp.clone());
        assert!(reg2.tenant_exists("acme"));
        assert_eq!(reg2.resolve(&key).unwrap().role, Role::Admin);
        let _ = std::fs::remove_file(&tmp);
    }
}
