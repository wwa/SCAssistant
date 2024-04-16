import os
import json
import traceback
from timeit import default_timer as timer
from Toolkit import SCAToolkit

def modelOne(toolkit, messages):
  ts_s = timer()
  print("Prompting...")
  res = toolkit.openai.chat.completions.create(
    model    = "gpt-4-turbo-preview",
    messages = messages,
    tools    = toolkit.toolMessage(),
    tool_choice = "auto"
  )
  ts_e = timer()
  print(f"... took {ts_e-ts_s}s")
  reason  = res.choices[0].finish_reason
  message = res.choices[0].message
  if reason == "stop":
    messages.append(json.loads(message.model_dump_json(exclude={'function_call', 'tool_calls'})))
    return reason, message.content, messages
  if reason == "tool_calls":
    # exclude because model_dump_json produces string Nones which can't be injested back
    messages.append(json.loads(message.model_dump_json(exclude={'function_call', 'content'})))
    for tc in message.tool_calls:
      if tc.type == "function":
        messages.append(toolkit.call(tc.id, tc.function))
  return reason,None,messages
def modelLoop(toolkit, history=[]):
  messages = [{
    "role": "system",
    "content": f"""
      Your job is to assist with side-channel research on physical hardware using ChipWhisperer. Do your best to produce device-under-test code, infer correct leakage models for it, compile it, collect power traces from it and execute trace analysis. If a run fails, attempt fixing problems automatically where possible and re-run. The toolkit contains multiple functions related to side-channels.
    """ + toolkit.toolPrompt()
    }] + sum(history, []) + [{"role":"user", "content":toolkit.userPrompt()}] + toolkit.fake('listTools')
  content = None
  while True:
    reason, content, messages = modelOne(toolkit, messages)
    if reason == "stop":
      break
  history.append(messages)
  return content, history

def mainLoop(toolkit, limit=10):
  history = []
  while True:
    try:
      prompt = toolkit.input()
      print(prompt)
      content, history = modelLoop(toolkit, history)
      history = history[:limit]
      print(content)
    except Exception as e:
      traceback.print_exc()
      pass

if __name__ == "__main__":
  toolkit = SCAToolkit()
  if not toolkit.openai:
    raise Exception('OpenAI API not initialized')
  mainLoop(toolkit)
