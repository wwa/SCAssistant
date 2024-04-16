import inspect
import json
import secrets
import traceback
from types import ModuleType
from openai import OpenAI
from dotenv import load_dotenv
from timeit import default_timer as timer

# Tool imports
import time
import os
import webbrowser
import threading
import pytesseract
import clipboard
import pyttsx3
import base64
#import pygetwindow
#import pyautogui
import serpapi
import arxiv
import urllib
import urllib.parse
from playsound import playsound
# import speech_recognition as sr
from PIL import ImageGrab, Image
from io import BytesIO

class AttrDict(dict):
  def __init__(self, *args, **kwargs):
    super(AttrDict, self).__init__(*args, **kwargs)
    self.__dict__ = self

def genToolspec(name, desc, args={}, reqs=[], **kwargs):
  # openAI tool_calls specification json
  # TODO: validate vs schema
  return {
    'type': 'function',
    'function': {
      'name': name,
      'description': desc,
      "parameters": {
        "type": "object",
        "properties": args,
        "required": reqs
      }
    }
  }
def toolspec(**kwargs):
  def decorator(func):
    if not hasattr(func, '_toolspec'):
      func._toolspec = AttrDict()
    source = kwargs.get('source')
    if source is None:
      try:
        source = inspect.getsource(func)
      except:
        pass
    func._toolspec = AttrDict({
      'state'    : 'enabled',
      'function' : func, 
      'spec'     : genToolspec(name = func.__name__, **kwargs),
      'source'   : source,
      'prompt'   : kwargs.get('prompt',"")
    })
    return func
  return decorator
def b64(img):
  if isinstance(img, Image.Image):
    with BytesIO() as buf:
      img.save(buf, format="PNG")
      return base64.b64encode(buf.getvalue()).decode('utf-8')
  with open(img, "rb") as f:
    return base64.b64encode(f.read()).decode('utf-8')

class Toolkit:
  # Contains toolkit barebones
  def __init__(self):
    self.data      = AttrDict()
    self.module    = ModuleType("DynaToolKit")
    self._toolspec = AttrDict()
    for name in dir(self):
      func = getattr(self, name)
      if not callable(func):
        continue
      if not hasattr(func, '_toolspec'):
        continue
      func._toolspec.function = func # overwrite with bound ref
      self._toolspec[name] = func._toolspec
    load_dotenv()
    if "OPENAI_API_KEY" in os.environ:
      self.openai = OpenAI()
    else:
      # model-assisted functions like addToolBySrc will be unavailable
      self.openai = None
  def toolspecBySrc(self, src, context=""):
    # Generates openAI tool_calls specifications from source code
    #   WARNING: model-generated, not bulletproof.
    if not self.openai:
      raise Exception("Model-assisted functions unavailable")
    res = self.openai.chat.completions.create(
      model    = "gpt-4-turbo-preview",
      messages = [{
        "role": "system",
        "content": f"""
          A Function description is an object describing a function and its arguments
          It consists of 3 elements:
            1. name: function name
            2. description: a short (2 sentences max) description of what the function does.
            3. arguments: an argument description
          An argument description is: {{name:<name>, type:<type>, description: <description>}} where description is a short (2 senteces max) description of the arguments purpose.
          <type> must be one of: number/integer/string
          Generate a function descriptions for each function in source code shown below.
          Answer in JSON {{functions: [{{name:<name>, description:<description>, args=[array of argument description]}},]}}
          <code>
          {src}
          </code>
          <context>
          {context}
          </context>
        """
      }],
      response_format={ "type": "json_object" }
    )
    descs = json.loads(json.loads(res.choices[0].message.model_dump_json())['content'])["functions"]
    tools = []
    for desc in descs:
      args = {}
      reqs = []
      for a in desc['args']:
        # args[a['name']] = {'type':a['type'], 'description':a['type']}
        # forcing type:string because models have weird ideas when generating types (e.g. type:url)
        args[a['name']] = {'type':'string', 'description':a['description']}
        reqs.append(a['name'])
      tools.append(genToolspec(desc['name'],desc['description'],args,reqs))
    return tools
  def addTool(self, func, spec, source=None, prompt=""):
    dec = toolspec(
      desc   = spec['function']['description'],
      args   = spec['function']['parameters']['properties'],
      reqs   = spec['function']['parameters']['required'],
      source = source,
      prompt = prompt
    )
    dec(func)
    self._toolspec[func.__name__] = func._toolspec
    return "{status: success}"
  def addToolByRef(self, func):
    # Registers a function by reference
    src  = inspect.getsource(func)
    spec = self.toolspecBySrc(src)[0]
    return self.addTool(func, spec, src)
  def toolPrompt(self):
    prompt = ""
    for k in self._toolspec:
      tool = self._toolspec[k]
      if tool.state == "enabled":
        prompt += tool.prompt
    return prompt
  def toolMessage(self):
    # Generates tool_calls table
    msgs = []
    for k in self._toolspec:
      tool = self._toolspec[k]
      if tool.state == "enabled":
        msgs.append(tool.spec)
    return msgs
  def call(self, cid, func):
    # Calls a tool.
    #   func is a message.tool_calls[i].function object
    ts_s = timer()
    print(f"Calling {func.name}")
    res = "Error: Unknown error."
    if func.name not in self._toolspec:
      res = "Error: Function not found."
    elif self._toolspec[func.name].state == "enabled":
      res = "Error: Function is disabled."
    try:
      args = json.loads(func.arguments)
      res = self._toolspec[func.name].function(**args)
    except Exception as e:
      # very important! most of the time model will correct itself if you let it know where it screwed up.
      res = f"Error: <backtrace>\n{traceback.format_exc()}\n</backtrace>"
      print(res)
      pass
    ts_e = timer()
    print(f"... took {ts_e-ts_s}s")
    return {
      "role": "tool", 
      "tool_call_id": cid,
      "name": func.name, 
      "content": f'{{"result": {str(res)}}}'
    }
  def fake(self,name,args='{}'):
    # Fake a tool call. Saves a model call while preserving context flow.
    # Use to pre-emptively inject data into history.
    func = AttrDict({'name':name, 'arguments':args})
    cid  = f"call_{secrets.token_urlsafe(24)}" # mimicking OpenAI IDs. Probably overkill.
    res  = self.call(cid,func)
    return [{
      'role': 'assistant',
      'tool_calls': [{
        'id': cid,
        'function': {
          'arguments': args,
          'name': name
        },
        'type': 'function'
      }],
    }, res]
  
  @toolspec(desc="Lists functions available in toolkit. Lists only disabled function by default.")
  def listTools(self, disabled=True):
    tools = []
    for name in self._toolspec:
      tool = self._toolspec[name]
      if tool.state == 'disabled' or not disabled:
        tools.append({'name': name, 'description': tool.spec['function']['description'], 'state':tool.state})
    return tools
  
  @toolspec(
    desc = "Toggles tool state: enabled/disabled. Disabled tools are not added to tool_calls, saving tokens",
    args = {
      "name":  {"type": "string", "description": "Python source code of functions to be added to toolkit"},
      "state": {"type": "string", "description": "One of: enabled/disabled"}
    },
    reqs = ["name","state"]
  )
  def toggleTool(self, name, state):
    #TODO: check if model thinks history is valid if a tool_call is removed
    if name not in self._toolspec:
      return f"{{status: error, error:{name} not found}}"
    self._toolspec[name].state = state
    return "{status: success}"
  
  @toolspec(
    desc = "Adds functions defined by Python source code to the toolkit. This should only be used if user explicitly asked to add a function to toolkit.",
    args = {"src": {"type": "string", "description": "Python source code of functions to be added to toolkit"}},
    reqs = ["src"]
  )
  def addToolBySrc(self, src):
    # Registers a function by source code
    logs  = ""
    code  = compile(src, self.module.__name__, 'exec')
    specs = self.toolspecBySrc(src)
    exec(code, self.module.__dict__)
    for spec in specs:
      print(spec)
      name = spec['function']['name']
      func = getattr(self.module, name)
      logs += self.addTool(func, spec, src)
    return logs
  
class BaseToolkit(Toolkit):
  # Contains basic user communication functions
  def __init__(self):
    super(BaseToolkit, self).__init__()
    self.serpapi  = serpapi.Client()
  def input(self):
    self.data.prompt = input()
    return self.data.prompt
  def userPrompt(self):
    return self.data.prompt
  
  @toolspec(
    desc = "Downloads file from URL. Returns local path of downloaded file.",
    args = {"url": {"type": "string", "description": "File to download"}},
    reqs = ["url"]
  )
  def download(url, filename=None):
    # downloads to tmp by default
    file, _ = urllib.request.urlretrieve(url, filename)
    return f"{{status: success, file={file}}}"
  
  @toolspec(
    desc = "Search the Internet. Returns top 10 results: {url, title, description}",
    args = {"phrase": {"type": "string",  "description": "Phrase to search for"},
            "limit":  {"type": "integer", "description": "Number of results. Default: 10"}},
    reqs = ["phrase"]
  )
  def webSearch(self, phrase, limit=10):
    res = self.serpapi.search({'engine': 'google','q': phrase})
    arr = [{'url': r['link'], 'title':r['title'], 'description': r['snippet']} for r in res['organic_results'][:limit]]
    return f"{{status: success, content:{json.dumps(arr)}}}"
  
  @toolspec(
    desc = "Search arxiv for publications. Returns {url:<permalink>, title:<title>, authors:<authors>, summary:<summary>}",
    args = {
      "query":  {"type": "string",  "description": "Arxiv query."},
      "limit":  {"type": "integer", "description": "Optional. Number of results. Default: 10"}
    },
    reqs = ["query"]
  )
  def arxivSearch(self, query, limit=10):
    print(f"{query}")
    client = arxiv.Client()
    res = client.results(arxiv.Search(
      query = query,
      max_results = limit
    ))
    entries = []
    for r in res:
      entries.append({'url': r.entry_id, 'title':r.title, 'authors':r.authors, 'summary':r.summary})
    return f"{{status: success, results:{entries}}}"
  
  @toolspec(
    desc = """ Run a research model. Reseach model can access files and run code.
      Multiple files can be passes in with "files" argument. Supports local files and Arxiv permalinks.
      Pass research_id to continue existing research. Leave empty to create new research thread.
    """,
    args = {
      "query":  {"type": "string", "description": "Research query."},
      "files":  {"type": "array",  "description": "Optional. Array of strings. List of files to include in research. Can be local files or Arxiv permalinks.", "items": {"type": "string"}},
      "research_id":  {"type": "string",  "description": "Optional. Research thread id. If empty, a new research thread will be created."},
    },
    reqs = ["query"],
    prompt = "When researching better results are achieved by reusing existing research thread and uploading multiple files to one thread."
  )
  def research(self, query, files=[], research_id=None):
    ass = None
    thr = None
    if not research_id:
      ass = self.openai.beta.assistants.create(
        instructions="""
          You are a research assistant.
          Your job is to process scientific papers.
          Display mathematical formulas using MathJax \[ markdown \] blocks.
        """,
        name  = "Echo research",
        tools = [{"type": "code_interpreter"}, {"type": "retrieval"}],
        model = "gpt-4-turbo-preview"
      )
      thr = self.openai.beta.threads.create(metadata={'aid':ass.id})
      print(f"New research context: {thr.id}")
    else:
      thr = self.openai.beta.threads.retrieve(research_id)
      ass = self.openai.beta.assistants.retrieve(thr.metadata['aid'])
      print(f"Loaded research context: {thr.id}")
    for file in files:
      print(f"Loading file: {file}")
      if not os.path.isfile(file):
        file = urllib.parse.urlparse(file).path.rsplit("/", 1)[-1]
        res  = arxiv.Search(id_list=[file])
        pdf  = next(res.results())
        file = pdf.download_pdf(dirpath="./downloads/")
      with open(file, "rb") as f:
        fid = self.openai.files.create(file = f, purpose = "assistants")
        self.openai.beta.assistants.files.create(assistant_id = ass.id, file_id = fid.id)
    print(f"Research query: {query}")
    ts_s = timer()
    msg  = self.openai.beta.threads.messages.create(thread_id = thr.id, role="user", content = query)
    run  = self.openai.beta.threads.runs.create(assistant_id = ass.id, thread_id = thr.id)
    #time.sleep(5) # FIXME?
    while run.status != "completed":
      time.sleep(1)
      run = self.openai.beta.threads.runs.retrieve(run_id = run.id, thread_id = run.thread_id)
    msg  = self.openai.beta.threads.messages.list(thread_id=run.thread_id,limit=1).data[0].content[0].text.value
    ts_e = timer()
    print(f"... took {ts_e-ts_s}s")
    return {'research_id': thr.id, 'message': msg}


import sys
sc_path = os.path.expanduser('~/sc-public/')
sys.path.append(os.path.join(sc_path, "pa-tools"))
from patools import ScopeTarget, TracesFileProxy_v2, CorrelationPowerAnalysis as CPA
from patools.traces import precompute_difftraces
import chipwhisperer as cw
import shutil
import uuid
import subprocess
import traceback
class SCAToolkit(BaseToolkit):
  def __init__(self):
    super(SCAToolkit, self).__init__()
    self.cw_path = os.path.expanduser('~/cw/chipwhisperer')
    self.sc_path = os.path.expanduser('~/sc-public/')
    self.sc_path = './data/'
    self.cw_platform = 'CW308_STM32F3'
    self.config_scope()
  def config_scope(self):
    self.st = ScopeTarget((), (cw.targets.SimpleSerial2,))
    self.st.target_clock = int(24e6)
    self.st.scope.default_setup()
    self.st.scope.gain.gain = 30
    self.st.scope.adc.samples = 24400
    self.st.scope.adc.offset = 0
    self.st.scope.clock.adc_src = "clkgen_x4"
    self.st.set_clock()
    self.st.reset_target()
      
  @toolspec(
    desc = "Prepares device under test (DUT) code. This is the on-device code to be anlyzed. Returns a unique DUT ID if successful.",
    args = {"code":  {"type": "string",  "description": """
        C code to be built. Contents of dut.c file to be compiled in a chipwhisperer-like build system.
        The code cannot generate 'main' function and must implement an entrypoint function with the following signature:
        void entrypoint(uint8_t* input, uint8_t* output, uint8_t* secret, uint32_t input_len, uint32_t secret_len)
        All symbols except 'entrypoint' should be static. The build system supports cryptographic libraries: mbedtls, wolfssl
        """},
      "lib":  {"type": "string",  "description": "libraries to link. Select one of: MBEDTLS, WOLFSSL, NONE. Default NONE."}
    },
    reqs = ["code"]
  )
  def cpa_build(self, code, lib="NONE"):
    print("cpa_build")
    print(code)
    print(lib)
    try:
      dutid   = uuid.uuid4()
      dutpath = os.path.join(self.sc_path, f"dut/{dutid}")
      tplpath = os.path.join(self.sc_path,  "dut/template")
      shutil.copytree(tplpath, dutpath)
      with open(os.path.join(dutpath, "dut.c"), 'w') as dut:
        dut.write(code)
      environment = os.environ.copy()
      environment.update({'CW_TARGET':lib, 'CW_PLATFORM':self.cw_platform, 'CW_PATH':self.cw_path})
      subprocess.run(('make',), env=environment, check=True, capture_output=True, cwd=dutpath)
      print(dutid)
      return dutid
    except subprocess.CalledProcessError as e:
      msg = f"{{'error': 'Build error', 'stdout': {e.stdout.decode()}, 'stderr': {e.stderr.decode()}}}"
      print(msg)
      return msg
  def cpa_flash(self, dutid):
    hexpath = os.path.join(self.sc_path, f"dut/{dutid}/simpleserial-dut-{self.cw_platform}.hex")
    self.st.reset_clock()
    prog = cw.programmers.STM32FProgrammer
    cw.program_target(self.st.scope, prog, hexpath)
    self.st.set_clock()
      
  @toolspec(
    desc = "Gathers power traces of DUT code from physical device. Returns a unique trace ID if successful.",
    args = {
      "dutid":  {"type": "string",  "description": "Unique DUT ID."},
      "trcnum": {"type": "integer", "description": "Number of traces. Must be between 1 and 5000. Optional, defaults to 200."},
      "secret": {"type": "string",  "description": "Hex-encoded value passed as 'secret' to the entrypoint function. Optional. If not provided, a random secret is generated."},
      "length": {"type": "integer", "description": "Byte length of the input to be generated for each invocation of the DUT entrypoint function. If the DUT code is a block cipher, the optimal value corresponds to block size. Must be between 0 and 255. Optional, defaults to 16."},
    },
    reqs = ["dutid"]
  )
  def cpa_trace(self, dutid, trcnum=200, secret=None, length=16):
    print(f"cpa_trace {dutid} {trcnum}")
    try:
      trcid = uuid.uuid4()
      self.cpa_flash(dutid)
      trcpath = os.path.join(self.sc_path, f"trc/{trcid}.tfp2")
      self.st.txi_bits = 8 * length
      self.st.txo_bits = 8 * length
      if secret is None:
        secret = os.urandom(length)
      else:
        secret = bytes.fromhex(secret)
      def gen(n):
        for _ in range(n): yield (secret, os.urandom(length))
      self.st.gather_for_generator(gen, trcpath, trcnum)
      return trcid
    except Exception as e:
      msg = f"{{'error': {e}}}"
      print(msg)
      print(traceback.format_exc())
      return msg
        
  @toolspec(
    desc = """
      Run Correlation Power Analysis on gathered traces.
      Arguments provide detailed description of the CPA task, using Python expressions. 
      All expressions must only use global variables defined within the `extra` argument. 
      """,
      # Additionally, MyreLabs PA Tools are imported into global namespace, includes function `hw(x)` that computes hamming weight of x for bytes and integers. 
      # The MyreLabs PA Tools source code can be found at https://gitlab.com/myrelabs/pa-tools.
    args = {
      "trcid": {"type": "string", "description": "Unique trace ID."},
      "extra": {"type": "string", "description": """Python code block containing required definitions. Should define imports, lookup tables, auxiliary functions etc. Examples:
          "import itertools"
          "from patools.victims.aes_tools import sbox"
          "def sbox(x): ..."
        For AES, common tables are provided by module `patools.victims.aes_tools`: `sbox`, `rev_sbox` (inverse sbox), `ft` (forward T-table), and `rt` (inverse T-table).
      """},
      "target":     {"type": "string", "description": """A single Python expression describing subset of bits or bytes of the `secret` variable to be target of analysis. Examples:
        "secret[0]"
        "secret[0] & 0xF"
        "secret[i]" where i is set in `extra` block.
      """},
      "hypothesis": {"type": "string", "description": """A single Python expression describing CPA leakage model. It is evaluated on three variables: input, output and `candidate` as returned by `candidates` generator. For example: "hw(func(input,output,candidate))" where func is defined by `extra` block.
      """},
      "candidates": {"type": "string", "description": """A single Python expression describing candidate generator. Examples: 
        "range(256)"
        "itertools.product(range(256), range(256))"
      """},
    },
    reqs = ["dutid"]
  )
  def cpa_analyze(self, trcid, extra, target, hypothesis, candidates):
    print("cpa_analyze")
    print(extra)
    print(target)
    print(hypothesis)
    print(candidates)
    try:
      trcpath = os.path.join(self.sc_path, f"trc/{trcid}.tfp2")
      with TracesFileProxy_v2.load(trcpath) as trc:
        _, difftraces, nvar_trace = precompute_difftraces(trc.traces)
        inouts = zip(trc.textins, trc.textouts)
        secret = trc.keys[0]
        _globals = {}
        exec('from patools import *', _globals)
        exec('from patools.utils.misc import *', _globals)
        exec(extra, _globals)
        model_func = lambda guess, inout: eval(hypothesis, _globals, {'input': inout[0], 'output': inout[1], 'candidate': guess})
        model_cand = eval(candidates, _globals)
        model_eval = lambda secr: eval(target, _globals, {'secret': secr})
        cpa  = CPA.single(model_func, difftraces, nvar_trace, inouts, model_cand)
        real = model_eval(secret)
      (rank, realcorr), = [(i, r[0]) for i, (g, r) in enumerate(cpa) if g == real]
      avgcorr = sum(r[0] for _, r in cpa) / len(cpa)
      maxcorr = cpa[0][1][0]
      mincorr = cpa[-1][1][0]
      res = {'real_candidate': real, 'rank': rank, 'corr': {'max': maxcorr, 'min': mincorr, 'avg': avgcorr, 'real': realcorr}}
      print(cpa)
      print(res)
      return cpa, res
    except Exception as e:
      msg = f"{{'error': {e}}}"
      print(msg)
      print(traceback.format_exc())
      return msg
      