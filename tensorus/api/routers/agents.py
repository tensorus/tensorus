from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query, status
from pydantic import BaseModel, Field

from ..context import (
    agent_registry,
    live_agents,
    _get_or_create_ingestion_agent,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class AgentInfo(BaseModel):
    id: str
    name: str
    description: str
    status: str
    config: Dict[str, Any]


class AgentStatus(AgentInfo):
    last_log_timestamp: Optional[float] = None


class AgentLogResponse(BaseModel):
    logs: List[str]


class AgentConfigPayload(BaseModel):
    config: Dict[str, Any]


@router.get("/agents", response_model=List[AgentInfo])
async def list_agents():
    agents_list = []
    for agent_id, details in agent_registry.items():
        if not isinstance(details, dict) or not all(k in details for k in ["name", "description", "config"]):
            logger.warning("Agent '%s' entry malformed", agent_id)
            continue
        status_val = "unknown"
        if agent_id == "ingestion":
            inst = live_agents.get("ingestion")
            status_val = inst.get_status() if inst else "stopped"
        elif "status" in details:
            status_val = details["status"]
        agents_list.append(
            AgentInfo(
                id=agent_id,
                name=details["name"],
                description=details["description"],
                status=status_val,
                config=details["config"],
            )
        )
    return agents_list


@router.get("/agents/{agent_id}/status", response_model=AgentStatus)
async def get_agent_status_api(agent_id: str = Path(...)):
    if agent_id not in agent_registry:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent '{agent_id}' not found.")
    details = agent_registry[agent_id]
    if agent_id == "ingestion":
        inst = _get_or_create_ingestion_agent()
        current_status = inst.get_status()
        last_log_ts = time.time()
    else:
        current_status = details.get("status", "unknown")
        last_log_ts = details.get("last_log_timestamp")
    return AgentStatus(
        id=agent_id,
        name=details["name"],
        description=details["description"],
        status=current_status,
        config=details["config"],
        last_log_timestamp=last_log_ts,
    )


@router.post("/agents/{agent_id}/start", response_model=AgentStatus, status_code=status.HTTP_202_ACCEPTED)
async def start_agent_api(agent_id: str = Path(...)):
    if agent_id not in agent_registry:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent '{agent_id}' not found.")
    if agent_id == "ingestion":
        inst = _get_or_create_ingestion_agent()
        if inst.get_status() == "running":
            return await get_agent_status_api(agent_id)
        inst.start()
        await asyncio.sleep(0.1)
        return await get_agent_status_api(agent_id)
    details = agent_registry[agent_id]
    if details.get("status") in {"running", "starting"}:
        return await get_agent_status_api(agent_id)
    details["status"] = "running"
    details["last_log_timestamp"] = time.time()
    return await get_agent_status_api(agent_id)


@router.post("/agents/{agent_id}/stop", response_model=AgentStatus, status_code=status.HTTP_202_ACCEPTED)
async def stop_agent_api(agent_id: str = Path(...)):
    if agent_id not in agent_registry:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent '{agent_id}' not found.")
    if agent_id == "ingestion":
        inst = live_agents.get("ingestion")
        if inst:
            inst.stop()
            await asyncio.sleep(0.1)
        return await get_agent_status_api(agent_id)
    details = agent_registry[agent_id]
    details["status"] = "stopped"
    details["last_log_timestamp"] = time.time()
    return await get_agent_status_api(agent_id)


@router.get("/agents/{agent_id}/logs", response_model=AgentLogResponse)
async def get_agent_logs_api(agent_id: str = Path(...), lines: int = Query(20, ge=1, le=1000)):
    if agent_id not in agent_registry:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent '{agent_id}' not found.")
    if agent_id == "ingestion":
        inst = _get_or_create_ingestion_agent()
        try:
            return AgentLogResponse(logs=inst.get_logs(max_lines=lines))
        except Exception as e:
            logger.exception("Error retrieving logs from DataIngestionAgent: %s", e)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error retrieving logs")
    # placeholder agents
    simulated = [f"{agent_id} log {i}" for i in range(lines)]
    agent_registry[agent_id]["last_log_timestamp"] = time.time()
    return AgentLogResponse(logs=simulated)


@router.get("/agents/{agent_id}/config", response_model=Dict[str, Any])
async def get_agent_config_api(agent_id: str = Path(...)):
    if agent_id not in agent_registry:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent '{agent_id}' not found.")
    return agent_registry[agent_id].get("config", {})


@router.post("/agents/{agent_id}/configure", response_model=Dict[str, Any])
async def configure_agent_api(agent_id: str = Path(...), payload: AgentConfigPayload = Body(...)):
    if agent_id not in agent_registry:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent '{agent_id}' not found.")
    new_config = payload.config
    agent_registry[agent_id]["config"] = new_config
    if agent_id == "ingestion" and "ingestion" in live_agents:
        inst = live_agents["ingestion"]
        if "source_directory" in new_config:
            inst.source_directory = new_config["source_directory"]
        if "dataset_name" in new_config:
            inst.dataset_name = new_config["dataset_name"]
        if "polling_interval_sec" in new_config:
            inst.polling_interval = new_config["polling_interval_sec"]
    return agent_registry[agent_id]["config"]
