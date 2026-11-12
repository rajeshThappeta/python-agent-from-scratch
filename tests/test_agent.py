"""Tests for the agent loop. They use a stand-in model, so Ollama is not needed."""

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SCRIPT = []  # the replies the stand-in model will give, in order
SEEN = []  # the messages the loop sent to the model on each step


def fake_chat(model, messages, tools):
    SEEN.append(list(messages))
    return SCRIPT.pop(0)


fake_ollama = types.ModuleType("ollama")
fake_ollama.chat = fake_chat
sys.modules["ollama"] = fake_ollama

import agent  # noqa: E402  (imported after the stand-in is in place)


def call(name, **arguments):
    return NS(function=NS(name=name, arguments=arguments))


def reply(content="", calls=None):
    return NS(message=NS(role="assistant", content=content, tool_calls=calls))


def tool_messages(messages):
    return [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"]


class AgentLoopTests(unittest.TestCase):
    def setUp(self):
        SCRIPT.clear()
        SEEN.clear()

    def test_two_tools_then_answer(self):
        SCRIPT.extend([
            reply(calls=[call("lookup_order", order_id="1042")]),
            reply(calls=[call("refund_policy")]),
            reply(content="Delivered on 2026-11-08. You can return it within 7 days."),
        ])
        answer = agent.run_agent("Where is order 1042, and can I return it?")
        self.assertTrue(answer.startswith("Delivered"))
        results = tool_messages(SEEN[-1])
        self.assertEqual([m["tool_name"] for m in results], ["lookup_order", "refund_policy"])
        self.assertIn("Delivered on 2026-11-08", results[0]["content"])

    def test_call_written_as_text_is_not_returned_as_the_answer(self):
        SCRIPT.extend([
            reply(content='lookup_order(order_id="1042")'),
            reply(calls=[call("lookup_order", order_id="1042")]),
            reply(content="It was delivered on 2026-11-08."),
        ])
        answer = agent.run_agent("Where is order 1042?")
        self.assertEqual(answer, "It was delivered on 2026-11-08.")
        self.assertEqual(SEEN[1][-1], {"role": "user", "content": "Use a real tool call."})

    def test_bad_argument_becomes_a_tool_error(self):
        SCRIPT.extend([
            reply(calls=[call("lookup_order", wrong_name="1042")]),
            reply(content="Sorry, I could not look that up."),
        ])
        agent.run_agent("Where is order 1042?")
        self.assertTrue(tool_messages(SEEN[-1])[0]["content"].startswith("Tool error:"))

    def test_unknown_tool_does_not_crash(self):
        SCRIPT.extend([
            reply(calls=[call("delete_everything")]),
            reply(content="I cannot do that."),
        ])
        agent.run_agent("Do something odd")
        self.assertTrue(tool_messages(SEEN[-1])[0]["content"].startswith("Tool error:"))

    def test_step_limit_stops_a_runaway_loop(self):
        SCRIPT.extend([reply(calls=[call("refund_policy")]) for _ in range(10)])
        answer = agent.run_agent("Loop forever", max_steps=5)
        self.assertEqual(answer, "Stopped: too many steps.")
        self.assertEqual(len(SEEN), 5)


if __name__ == "__main__":
    unittest.main()
