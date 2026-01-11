# agent_fun.py
import asyncio, json, sys
from typing import Dict, Any, List
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from ollama import chat  # pip install ollama

SYSTEM = (
    "You are a cheerful weekend helper.\n"
    "You MUST follow this rule strictly:\n"
    "- Always respond with VALID JSON\n"
    "- NEVER include explanations, markdown, or text outside JSON\n\n"

    "If you need to call a tool, respond exactly like:\n"
    '{"action":"tool_name","args":{...}}\n\n'

    "If you can answer directly, respond exactly like:\n"
    '{"action":"final","answer":"..."}\n\n'

    "If unsure, choose the best action.\n"
)


# def llm_json(messages: List[Dict[str, str]]) -> Dict[str, Any]:
#     resp = chat(model="mistral:7b", messages=messages, options={"temperature": 0.2})
#     txt = resp["message"]["content"].strip()
    
#     # Extract ONLY the first JSON object
#     if "{" in txt:
#         start = txt.find("{")
#         # Find the matching closing brace for the first object
#         brace_count = 0
#         end = start
#         for i in range(start, len(txt)):
#             if txt[i] == "{":
#                 brace_count += 1
#             elif txt[i] == "}":
#                 brace_count -= 1
#                 if brace_count == 0:
#                     end = i + 1
#                     break
#         txt = txt[start:end]
    
#     try:
#         return json.loads(txt)
#     except Exception as e:
#         print(f"JSON parse error: {e}")
#         print(f"Raw response: {txt}")
#         # Ask LLM to give just ONE action
#         fix = chat(model="mistral:7b",
#                    messages=[{"role": "system", "content": "Return ONLY ONE valid JSON action object. Pick the FIRST action needed."},
#                              {"role": "user", "content": f"Extract just the FIRST action from: {txt}"}],
#                    options={"temperature": 0})
#         fixed_txt = fix["message"]["content"].strip()
#         if "{" in fixed_txt:
#             start = fixed_txt.find("{")
#             brace_count = 0
#             end = start
#             for i in range(start, len(fixed_txt)):
#                 if fixed_txt[i] == "{":
#                     brace_count += 1
#                 elif fixed_txt[i] == "}":
#                     brace_count -= 1
#                     if brace_count == 0:
#                         end = i + 1
#                         break
#             fixed_txt = fixed_txt[start:end]
#         return json.loads(fixed_txt)

def llm_json(messages):
    resp = chat(
        model="mistral:7b",
        messages=messages,
        options={"temperature": 0}
    )
    txt = resp["message"]["content"].strip()

    try:
        data = json.loads(txt)
    except Exception:
        # HARD repair: force correct schema
        fix = chat(
            model="mistral:7b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return ONLY valid JSON in this exact format:\n"
                        '{"action":"final","answer":"..."}\n'
                        "or\n"
                        '{"action":"tool_name","args":{...}}\n'
                        "Do not add text."
                    )
                },
                {"role": "user", "content": txt}
            ],
            options={"temperature": 0}
        )
        data = json.loads(fix["message"]["content"])

    # 🚨 FINAL SAFETY CHECK
    if not isinstance(data, dict):
        return {"action": "final", "answer": str(data)}

    if not isinstance(data.get("action"), str):
        return {"action": "final", "answer": "Something went wrong, please retry."}

    return data


async def main():
    server_path = sys.argv[1] if len(sys.argv) > 1 else "server_fun.py"
    exit_stack = AsyncExitStack()
    stdio = await exit_stack.enter_async_context(
        stdio_client(StdioServerParameters(command="python", args=[server_path]))
    )
    r_in, w_out = stdio
    session = await exit_stack.enter_async_context(ClientSession(r_in, w_out))
    await session.initialize()

    tools = (await session.list_tools()).tools
    tool_index = {t.name: t for t in tools}
    print("Connected tools:", list(tool_index.keys()))

    history = [{"role": "system", "content": SYSTEM}]
    try:
        while True:
            user = input("You: ").strip()
            if not user or user.lower() in {"exit","quit"}: break
            history.append({"role": "user", "content": user})

            for step in range(8):  # allow enough steps for multiple tool calls
                try:
                    decision = llm_json(history)
                except Exception as e:
                    print(f"Error getting decision: {e}")
                    history.append({"role":"assistant","content": f"(error: {e})"})
                    continue
                    
                if decision.get("action") == "final":
                    answer = decision.get("answer","")
                    # one-shot reflection
                    try:
                        reflect = chat(model="mistral:7b",
                                       messages=[{"role":"system","content":"Check for mistakes or missing tool calls. If fine, reply 'looks good'; else give corrected answer."},
                                                 {"role":"user","content": answer}],
                                       options={"temperature": 0})
                        reflection = reflect["message"]["content"].strip().lower()

                        if reflection != "looks good":
                            # Only replace if reflection actually contains a better final answer
                            if len(reflection) < len(answer):
                                answer = reflect["message"]["content"]

                        # if reflect["message"]["content"].strip().lower() != "looks good":
                        #     answer = reflect["message"]["content"]
                    except:
                        pass  # skip reflection if it fails
                    print("Agent:", answer)
                    history.append({"role":"assistant","content": answer})
                    break

                tname = decision.get("action")
                args = decision.get("args", {})
                if not tname or tname not in tool_index:
                    history.append({"role":"assistant","content": f"(unknown tool {tname})"})
                    continue

                print(f"[Calling tool: {tname} with {args}]")
                result = await session.call_tool(tname, args)
                payload = result.content[0].text if result.content else result.model_dump_json()
                history.append({"role":"assistant","content": f"[tool:{tname}] {payload}"})
    finally:
        await exit_stack.aclose()

if __name__ == "__main__":
    asyncio.run(main())
