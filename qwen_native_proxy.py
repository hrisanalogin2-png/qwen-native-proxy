import os, json, uuid, hashlib, time, logging
from typing import Optional
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("proxy")

QWEN_API = os.environ.get("QWEN_API", "https://chat.qwen.ai/api/v2")
QWEN_TOKEN = os.environ.get("QWEN_TOKEN", "")

QWEN_HEADERS = {
    "Authorization": f"Bearer {QWEN_TOKEN}",
    "Content-Type": "application/json",
    "Origin": "https://qwen.ai",
    "Referer": "https://qwen.ai/",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
}

http = httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=15.0), verify=False)

class Session:
    __slots__ = ("chat_id", "parent_id", "tool_calls_map", "created")
    def __init__(self, chat_id: str):
        self.chat_id = chat_id
        self.parent_id: Optional[str] = None
        self.tool_calls_map: dict[str, dict] = {}
        self.created = time.time()

_sessions: dict[str, Session] = {}
_sess_lock = __import__("asyncio").Lock()

async def get_session(sid: str, model: str) -> Session:
    async with _sess_lock:
        if sid not in _sessions:
            r = await http.post(f"{QWEN_API}/chats/new", headers=QWEN_HEADERS,
                json={"model": model}, timeout=httpx.Timeout(30.0, connect=10.0))
            r.raise_for_status()
            d = r.json()["data"]
            _sessions[sid] = Session(d["id"])
            log.info(f"new session={sid[:8]} chat={d['id'][:8]} model={model}")
        return _sessions[sid]

BASE_FC = {"thinking_enabled": False, "output_schema": "phase", "auto_search": False}
TOOL_FC = {"thinking_enabled": False, "output_schema": "phase", "auto_search": False}

def _check_qwen_error(raw: str):
    if raw.startswith("{"):
        try:
            d = json.loads(raw)
            if not d.get("success", True):
                code = d.get("data", {}).get("code", "unknown")
                msg = d.get("data", {}).get("message", d.get("message", ""))
                return f"[Qwen API Error: {code}] {msg}"
        except (json.JSONDecodeError, AttributeError):
            pass
    return None

def tools_to_local_mcp(tools: list) -> dict:
    if not tools:
        return {}
    result = {}
    for fn in tools:
        f = fn.get("function", fn) if isinstance(fn, dict) else fn
        name = f.get("name", "")
        if not name:
            continue
        result[name] = {
            "description": f.get("description", ""),
            "input_schema": f.get("parameters", {"type": "object", "properties": {}}),
            "type": "local_mcp",
            "runtime": True,
        }
    return {"opencode_tools": result} if result else {}

def parse_qwen_delta(data: dict, buffer: dict) -> dict:
    out = {"finish_reason": None, "tool_calls": None}
    choices = data.get("choices", [])
    if not choices:
        return out
    delta = choices[0].get("delta", {})
    if not delta:
        return out

    phase = delta.get("phase", "")
    tokens = delta.get("content", "") or delta.get("tokens", "")
    extra = delta.get("extra", {})
    status = delta.get("status", "")

    if not phase:
        if tokens:
            out["content"] = tokens
        return out

    if phase == "think":
        buffer["think"] = True
        if tokens:
            out["content"] = tokens
        return out

    if phase == "local_tool":
        mcp_name = delta.get("mcp_name", "")
        t_name = delta.get("tool_name", "")

        if status == "typing":
            tid = f"call_{hashlib.sha256(f'{mcp_name}:{t_name}'.encode()).hexdigest()[:16]}"
            buffer.update({"id": tid, "name": t_name, "phase": "typing", "mcp_server": mcp_name})
            out["tool_calls"] = [{
                "index": 0, "id": tid, "type": "function",
                "function": {"name": t_name, "arguments": ""},
            }]
            return out

        if status == "finished":
            local_mcp_extra = extra.get("local_mcp", {})
            params_str = "{}"
            tool_name = t_name
            mcp_server = mcp_name

            for sname, tools_list in local_mcp_extra.items():
                for tool_entry in tools_list:
                    tn = tool_entry.get("tool_name", "")
                    if tn:
                        tool_name = tn
                        mcp_server = sname
                    p = tool_entry.get("params", {})
                    if p:
                        params_str = json.dumps(p)

            tid = buffer.get("id",
                f"call_{hashlib.sha256(f'{mcp_server}:{tool_name}'.encode()).hexdigest()[:16]}")
            buffer.update({"id": tid, "name": tool_name, "mcp_server": mcp_server, "phase": "done"})
            out["tool_calls"] = [{
                "index": 0, "id": tid, "type": "function",
                "function": {"name": tool_name, "arguments": params_str},
            }]
            out["finish_reason"] = "tool_calls"
            out["done"] = True
            return out

    if phase == "usage":
        return {"usage": data.get("extra", {}).get("usage", {})}

    if phase == "search_queries":
        out["content"] = ""
        return out

    if tokens:
        out["content"] = tokens

    return out

    if phase == "think":
        buffer["think"] = True
        if tokens:
            out["content"] = tokens
        return out

    if phase == "local_tool":
        status = delta.get("status", "") or extra.get("status", "")
        mcp_name = delta.get("mcp_name", "") or extra.get("mcp_name", "")
        t_name = delta.get("tool_name", "") or extra.get("tool_name", "")
        params_json = delta.get("params", "{}") or extra.get("params", "{}")

        if status == "typing":
            tid = f"call_{hashlib.sha256(f'{mcp_name}:{t_name}'.encode()).hexdigest()[:16]}"
            buffer.update({"id": tid, "name": t_name, "phase": "typing", "mcp_server": mcp_name})
            out["tool_calls"] = [{
                "index": 0, "id": tid, "type": "function",
                "function": {"name": t_name, "arguments": ""},
            }]
            return out

        if status == "finished":
            if buffer.get("phase") == "typing":
                buffer["phase"] = "done"
                t_name = buffer.get("name", t_name)
                mcp_server = buffer.get("mcp_server", mcp_name)
                tid = buffer.get("id", f"call_{hashlib.sha256(f'{mcp_server}:{t_name}'.encode()).hexdigest()[:16]}")
                args_str = params_json if isinstance(params_json, str) else json.dumps(params_json)
                try:
                    json.loads(args_str)
                except (json.JSONDecodeError, TypeError):
                    args_str = "{}"
                out["tool_calls"] = [{
                    "index": 0, "id": tid, "type": "function",
                    "function": {"name": t_name, "arguments": args_str},
                }]
                out["finish_reason"] = "tool_calls"
                out["done"] = True
            return out

    if phase == "usage":
        return {"usage": delta.get("extra", {}).get("usage", {})}

    if phase == "search_queries":
        out["content"] = ""
        return out

    return out

def get_session_id(headers, msgs: list) -> str:
    sid = headers.get("x-session-id", "") or headers.get("X-Session-Id", "")
    if sid:
        return sid
    for msg in msgs:
        if msg.get("role") == "user":
            key = str(msg.get("content", ""))[:64]
            return hashlib.md5(key.encode()).hexdigest()[:16]
    return str(uuid.uuid4())

def _qwen_msg(role: str, content: str, model: str, parent_id: str = None, fc: dict = None):
    msg = {
        "role": role,
        "content": content or "",
        "fid": str(uuid.uuid4()),
        "childrenIds": [str(uuid.uuid4())],
        "timestamp": int(time.time() * 1000),
        "models": [model],
        "chat_type": "t2t",
        "feature_config": fc or dict(BASE_FC),
        "extra": {"meta": {"subChatType": "t2t"}},
        "user_action": "chat",
        "files": [],
    }
    if parent_id:
        msg["parentId"] = parent_id
    return msg

def _build_tool_result_msg(msgs: list, model: str, session: Session) -> list:
    tc = None
    result = ""
    for m in msgs:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            tc = m["tool_calls"][0]
        if m.get("role") == "tool":
            result = m.get("content", "") or ""
    if not tc:
        return [_qwen_msg("user", result, model)]
    tc_id = tc.get("id", "")
    info = session.tool_calls_map.pop(tc_id, None)
    server = info["server"] if info else "opencode_tools"
    tname = info["tool_name"] if info else tc.get("function", {}).get("name", "tool")
    results_map = {server: [{tname: result}]}
    parent_id = session.parent_id
    msg = {
        "role": "function",
        "content": results_map,
        "fid": str(uuid.uuid4()),
        "childrenIds": [str(uuid.uuid4())],
        "parentId": parent_id,
        "timestamp": int(time.time() * 1000),
        "models": [model],
        "chat_type": "t2t",
        "feature_config": {"thinking_enabled": False, "output_schema": "phase"},
        "extra": {"meta": {"subChatType": "t2t"}},
        "user_action": "chat",
        "files": [],
    }
    return [msg]

def openai_to_qwen_msgs(msgs: list, model: str, tools: list, session: Session) -> list:
    last = msgs[-1] if msgs else {}

    if last.get("role") == "tool":
        return _build_tool_result_msg(msgs, model, session)

    local_mcp = tools_to_local_mcp(tools)
    has_tools = bool(local_mcp)

    def make_fc():
        fc = dict(TOOL_FC) if has_tools else dict(BASE_FC)
        if has_tools:
            fc["local_mcp"] = local_mcp
            fc["mcp"] = list(local_mcp.keys())
        return fc

    if session.parent_id:
        last_content = last.get("content", "") or ""
        if last.get("role") == "user":
            return [_qwen_msg("user", last_content, model, parent_id=session.parent_id, fc=make_fc())]
        if last.get("role") == "assistant":
            return [_qwen_msg("assistant", last_content, model, parent_id=session.parent_id)]

    qwen_msgs = []
    sys_buf = ""
    for msg in msgs[:-1]:
        role = msg.get("role", "user")
        content = msg.get("content", "") or ""
        if role == "system":
            sys_buf = content
            continue
        qwen_msgs.append(_qwen_msg(role, content, model))
    last_role = last.get("role", "user")
    last_content = last.get("content", "") or ""
    if sys_buf and last_role == "user":
        last_content = sys_buf + "\n\n" + last_content
    qwen_msgs.append(_qwen_msg(last_role, last_content, model, parent_id=session.parent_id, fc=make_fc()))
    return qwen_msgs

async def stream_qwen(chat_id: str, model: str, msgs: list, parent_id: str, openai_model: str, session: Session):
    payload = {
        "chat_id": chat_id, "model": model, "chat_mode": "normal",
        "messages": msgs, "stream": True, "version": "2.1",
        "incremental_output": True,
    }
    if parent_id:
        payload["parent_id"] = parent_id

    log.info(f"Qwen payload messages[0] feature_config={json.dumps(msgs[0].get('feature_config',{}), ensure_ascii=False)[:500]}")
    log.info(f"Qwen has local_mcp={'local_mcp' in str(msgs[0].get('feature_config',{}))}")
    
    buffer = {}
    finish_reason = None
    tool_calls_done = False
    response_id = None
    sent_done = False

    try:
        async with http.stream("POST",
            f"{QWEN_API}/chat/completions?chat_id={chat_id}",
            headers=QWEN_HEADERS, json=payload,
            timeout=httpx.Timeout(180.0, connect=15.0)) as resp:
            resp.raise_for_status()
            buf = ""
            async for raw_bytes in resp.aiter_bytes():
                buf += raw_bytes.decode(errors="replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    if not line.startswith("data: "):
                        stripped = line.strip()
                        if stripped.startswith("{"):
                            err = _check_qwen_error(stripped)
                            if err and not sent_done:
                                yield f"data: {json.dumps({'id': f'chatcmpl-{uuid.uuid4().hex[:12]}', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': openai_model, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': err}, 'finish_reason': 'stop'}]})}\n\n"
                                yield "data: [DONE]\n\n"
                                sent_done = True
                                return
                        continue
                    raw = line[6:].strip()
                    if not raw or raw == "[DONE]":
                        if not sent_done:
                            yield "data: [DONE]\n\n"
                            sent_done = True
                        return

                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    if "response.created" in data:
                        response_id = data["response.created"].get("response_id")
                        if response_id:
                            session.parent_id = response_id
                        continue

                    if "response.stopped" in data:
                        fr = data["response.stopped"].get("finish_reason", "stop")
                        if fr == "tool_calls":
                            finish_reason = "tool_calls"
                        else:
                            finish_reason = fr
                        continue

                    choices = data.get("choices", [])
                    if not choices:
                        continue

                    parsed = parse_qwen_delta(data, buffer)
                    if parsed.get("tool_calls") and not tool_calls_done:
                        tc = parsed["tool_calls"]
                        if tc and tc[0]["function"]["name"]:
                            tname = tc[0]["function"]["name"]
                            for t in tc:
                                tid = t.get("id", "")
                                if tid:
                                    session.tool_calls_map[tid] = {
                                        "name": tname,
                                        "tool_name": tname,
                                        "server": "opencode_tools",
                                    }
                            if tname == "bash" and tc[0]["function"]["arguments"]:
                                try:
                                    a = json.loads(tc[0]["function"]["arguments"])
                                    if isinstance(a, dict) and "command" in a and isinstance(a["command"], str):
                                        cmd = a["command"].strip()
                                        if not cmd.rstrip().endswith("&"):
                                            a["command"] = cmd + " &"
                                            tc[0]["function"]["arguments"] = json.dumps(a)
                                except Exception:
                                    pass
                            out = {
                                "id": f"chatcmpl-{response_id or uuid.uuid4().hex[:12]}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": openai_model,
                                "choices": [{"index": 0, "delta": {"role": "assistant", "content": None,
                                    "tool_calls": tc}, "finish_reason": None}],
                            }
                            yield f"data: {json.dumps(out, ensure_ascii=False)}\n\n"
                        if parsed.get("finish_reason") == "tool_calls":
                            finish_reason = "tool_calls"
                            tool_calls_done = True
                            out = {
                                "id": f"chatcmpl-{response_id or uuid.uuid4().hex[:12]}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": openai_model,
                                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                            }
                            yield f"data: {json.dumps(out, ensure_ascii=False)}\n\n"
                            yield "data: [DONE]\n\n"
                            sent_done = True
                            return
                        continue

                    if parsed.get("content"):
                        out = {
                            "id": f"chatcmpl-{response_id or uuid.uuid4().hex[:12]}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": openai_model,
                            "choices": [{"index": 0, "delta": {"content": parsed["content"]}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(out, ensure_ascii=False)}\n\n"

            if finish_reason and not sent_done:
                out = {
                    "id": f"chatcmpl-{response_id or uuid.uuid4().hex[:12]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": openai_model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                }
                yield f"data: {json.dumps(out, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                sent_done = True
    except Exception as e:
        log.warning(f"stream error: {e}")
        if not sent_done:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

async def non_stream_qwen(chat_id: str, model: str, msgs: list, parent_id: str, openai_model: str, session: Session) -> dict:
    payload = {
        "chat_id": chat_id, "model": model, "chat_mode": "normal",
        "messages": msgs, "stream": True, "version": "2.1",
        "incremental_output": True,
    }
    if parent_id:
        payload["parent_id"] = parent_id

    content_parts = []
    reasoning = []
    response_id = None
    finish_reason = "stop"
    buffer = {}
    tool_calls = None
    tool_calls_done = False
    error_content = None

    try:
        async with http.stream("POST",
            f"{QWEN_API}/chat/completions?chat_id={chat_id}",
            headers=QWEN_HEADERS, json=payload,
            timeout=httpx.Timeout(180.0, connect=15.0)) as resp:
            resp.raise_for_status()
            buf = ""
            async for raw_bytes in resp.aiter_bytes():
                buf += raw_bytes.decode(errors="replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    if not line.startswith("data: "):
                        stripped = line.strip()
                        if stripped.startswith("{"):
                            err = _check_qwen_error(stripped)
                            if err:
                                log.warning(f"Qwen API error: {err}")
                                error_content = err
                                break
                        continue
                    raw = line[6:].strip()
                    if not raw or raw == "[DONE]":
                        break

                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    if "response.created" in data:
                        response_id = data["response.created"].get("response_id")
                        if response_id:
                            session.parent_id = response_id
                        continue

                    if "response.stopped" in data:
                        fr = data["response.stopped"].get("finish_reason", "stop")
                        if fr == "tool_calls":
                            finish_reason = "tool_calls"
                        else:
                            finish_reason = fr
                        break

                    choices = data.get("choices", [])
                    if not choices:
                        continue

                    parsed = parse_qwen_delta(data, buffer)
                    if parsed.get("content"):
                        content_parts.append(parsed["content"])

                    if parsed.get("tool_calls") and not tool_calls_done:
                        tc = parsed["tool_calls"]
                        if tc and tc[0].get("function", {}).get("name"):
                            tool_calls = tc
                            tname = tc[0]["function"]["name"]
                            for t in tc:
                                tid = t.get("id", "")
                                if tid:
                                    session.tool_calls_map[tid] = {
                                        "name": tname,
                                        "tool_name": tname,
                                        "server": "opencode_tools",
                                    }
                            if tname == "bash" and tc[0]["function"]["arguments"]:
                                try:
                                    a = json.loads(tc[0]["function"]["arguments"])
                                    if isinstance(a, dict) and "command" in a and isinstance(a["command"], str):
                                        cmd = a["command"].strip()
                                        if not cmd.rstrip().endswith("&"):
                                            a["command"] = cmd + " &"
                                            tc[0]["function"]["arguments"] = json.dumps(a)
                                except Exception:
                                    pass
                        if parsed.get("finish_reason") == "tool_calls":
                            tool_calls_done = True
                            finish_reason = "tool_calls"
                            break
    except Exception as e:
        log.warning(f"non_stream error: {e}")

    content = "".join(content_parts).strip()
    if error_content:
        content = error_content
    msg = {"role": "assistant", "content": content or None}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "id": f"chatcmpl-{response_id or uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": openai_model,
        "choices": [{"index": 0, "message": msg, "finish_reason": finish_reason or "stop"}],
    }

app = FastAPI()

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    t0 = time.time()
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid body"}, 400)
    msgs = body.get("messages", [])
    model = body.get("model", "qwen3.7-max")
    tools = body.get("tools", [])
    is_stream = body.get("stream", False)
    sid = get_session_id(request.headers, msgs)
    roles = [m.get("role") for m in msgs]
    has_tool_result = len(msgs) >= 2 and msgs[-1].get("role") == "tool"

    session = await get_session(sid, model)

    is_new_conversation = all(r in ("system", "user") for r in roles)
    if is_new_conversation and session.parent_id is not None:
        log.info(f"new conversation detected, resetting parent_id for session={sid[:8]}")
        session.parent_id = None
        session.tool_calls_map.clear()

    qwen_msgs = openai_to_qwen_msgs(msgs, model, tools, session)
    pid = session.parent_id

    local_mcp_check = tools_to_local_mcp(tools)
    log.info(f"sid={sid[:8]} chat={session.chat_id[:8]} parent={str(pid)[:8] if pid else 'new'} "
             f"turn={len(_sessions):d} tools={bool(tools)} tool_result={has_tool_result} "
             f"qwen_msgs={len(qwen_msgs)} mcp={bool(local_mcp_check)}")
    payload_json = json.dumps(qwen_msgs[0] if qwen_msgs else {}, ensure_ascii=False)
    log.info(f"msg_preview={payload_json[:800]}")

    if is_stream:
        return StreamingResponse(
            stream_qwen(session.chat_id, model, qwen_msgs, pid, model, session),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    result = await non_stream_qwen(session.chat_id, model, qwen_msgs, pid, model, session)
    return JSONResponse(result)

@app.get("/health")
async def health():
    return {"status": "ok", "sessions": len(_sessions)}

@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": "qwen3.7-max", "object": "model", "created": int(time.time()), "owned_by": "qwen"}]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
