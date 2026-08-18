# prompt caching and per-session question cap

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import research_assistant.agent.agent_loop as agent_loop  # noqa: E402
from research_assistant.agent.agent_loop import ResearchAgent  # noqa: E402


class FakeTextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class FakeMessages:
    """Records every request it receives and always answers immediately
    (stop_reason == 'end_turn'), so tests don't need to simulate tool use."""

    def __init__(self):
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(
            content=[FakeTextBlock(f"answer #{len(self.requests)}")],
            stop_reason="end_turn",
        )


def _make_agent(max_questions=5):
    agent = ResearchAgent.__new__(ResearchAgent)  # bypass __init__'s real Anthropic() call
    agent._client = SimpleNamespace(messages=FakeMessages())
    agent._model = "claude-sonnet-5"
    agent._max_tool_iterations = 8
    agent._verbose = False
    agent._history = []
    agent._max_questions = max_questions
    agent._questions_asked = 0
    return agent


def test_system_and_tools_carry_cache_control():
    agent = _make_agent()
    agent.ask("what is reinforcement learning?")
    req = agent._client.messages.create.__self__.requests[0]
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert req["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    print("OK  system prompt and tools carry cache_control breakpoints")


def test_message_breakpoint_moves_and_stays_singular():
    agent = _make_agent()
    agent.ask("first question")
    agent.ask("second question")
    requests = agent._client.messages.create.__self__.requests

    def count_message_breakpoints(req):
        n = 0
        for msg in req["messages"]:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "cache_control" in block:
                        n += 1
        return n

    for req in requests:
        assert count_message_breakpoints(req) == 1, "expected exactly one moving breakpoint in messages"

    # The breakpoint in the second request should sit on the *second*
    # question, i.e. it moved forward rather than staying on the first.
    second_req_messages = requests[-1]["messages"]
    last_user_msg = [m for m in second_req_messages if m["role"] == "user"][-1]
    assert "second question" in str(last_user_msg["content"])
    assert last_user_msg["content"][-1].get("cache_control") == {"type": "ephemeral"}
    print("OK  message-history cache breakpoint moves forward each turn, never duplicates")


def test_total_breakpoints_within_api_limit():
    agent = _make_agent()
    for i in range(4):
        agent.ask(f"question {i}")
    req = agent._client.messages.create.__self__.requests[-1]

    total = 0
    if isinstance(req["system"], list):
        total += sum(1 for b in req["system"] if "cache_control" in b)
    total += sum(1 for t in req["tools"] if "cache_control" in t)
    for msg in req["messages"]:
        content = msg.get("content")
        if isinstance(content, list):
            total += sum(1 for b in content if isinstance(b, dict) and "cache_control" in b)

    assert total <= 4, f"exceeded Anthropic's 4-cache-breakpoint-per-request limit, got {total}"
    print(f"OK  total cache breakpoints ({total}) stay within the API's limit of 4, even after many turns")


def test_question_cap_blocks_without_calling_api():
    agent = _make_agent(max_questions=2)
    agent.ask("q1")
    agent.ask("q2")
    calls_before = len(agent._client.messages.create.__self__.requests)
    answer = agent.ask("q3 should be blocked")
    calls_after = len(agent._client.messages.create.__self__.requests)

    assert calls_after == calls_before, "the capped question must not call the API at all"
    assert "limit" in answer.lower()
    assert agent.questions_remaining() == 0
    print("OK  question cap blocks the API call entirely once exhausted (saves cost)")


def test_cap_disabled_when_zero():
    agent = _make_agent(max_questions=0)
    for i in range(10):
        agent.ask(f"question {i}")
    assert len(agent._client.messages.create.__self__.requests) == 10
    assert agent.questions_remaining() is None
    print("OK  DEMO_MAX_QUESTIONS=0 disables the cap entirely")


if __name__ == "__main__":
    test_system_and_tools_carry_cache_control()
    test_message_breakpoint_moves_and_stays_singular()
    test_total_breakpoints_within_api_limit()
    test_question_cap_blocks_without_calling_api()
    test_cap_disabled_when_zero()
    print("\nAll agent-loop tests passed.")