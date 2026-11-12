# Python Agent From Scratch

An AI agent in about 30 lines of plain Python. No framework. A local model through Ollama.

This is the code behind my carousel "An AI agent in about 30 lines of plain Python".
Write the loop once, before you pick a framework. After that, any framework is easier to read.

## What it does

A small order-status helper. A customer asks:

> Where is order 1042, and can I return it?

The model decides which tool to use:

| Tool | What it does |
|---|---|
| `lookup_order(order_id)` | Returns the status and date of one order |
| `refund_policy()` | Returns the return and refund rules |

The order data and the policy text are sample data.

## The idea

An agent is a loop.

```
question
   |
   v
 model --- asks for a tool? --- no ---> final answer
   |
  yes
   |
   v
 your code runs the tool
   |
   v
 result goes back to the model ---> (repeat)
```

The model never runs your code. It only asks. Your code runs the tool and hands the result back.

## Setup

You need Python 3.9 or newer and [Ollama](https://ollama.com).

```bash
# 1. Pull a model that supports tool calling
ollama pull qwen3:14b

# 2. Install the Python package
pip install -r requirements.txt

# 3. Run the agent
python agent.py
python agent.py "What is your return policy?"
```

The answer is written by the model, so the wording changes from run to run.
To try another model, change the `model` argument of `run_agent` in `agent.py`.

## How the code is built, in four steps

1. **Tools are plain functions.** The docstring is what the model reads to decide when to use the tool.
2. **Hand the tools to the model.** The `ollama` package reads each function's signature and docstring and builds the tool description.
3. **The loop.** Ask the model, run the tools it asked for, send the results back, repeat.
4. **Two guards.**
   - *A call written as text.* A model sometimes describes a tool call in its reply instead of making one. With a smaller model (qwen3:8b) I saw this in my own build. If the reply has no tool calls but names a tool, the loop asks again.
   - *A loop that never ends.* `max_steps` stops the loop after a fixed number of passes.

A tool that fails does not crash the agent. The error text goes back to the model, which can recover.

## Run the tests

```bash
python -m unittest discover -s tests -v
```

The tests use a stand-in model, so they run without Ollama. They check the loop, not the model:
a normal two-tool run, a call written as text, a bad argument, an unknown tool and a runaway loop.

## Limits

- Sample data only. This is a teaching example, not production code.
- No memory between questions. Each run starts fresh.
- Local models behave differently from run to run. The tests cover the loop logic, not the model's choices.
- The text-call guard is a simple name match. A normal answer that happens to mention a tool name would trigger a retry.

## Related

- [job-ready-agent](https://github.com/rajeshThappeta/job-ready-agent): a larger agent in JavaScript with LangChain.js, memory and human approval.

## Author

Rajesh T, AI Powered Full Stack Educator, MastersCoding AI Labs.
[LinkedIn](https://www.linkedin.com/in/rajesh-t)
