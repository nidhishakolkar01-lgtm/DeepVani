"""
Unit tests for DeepVani AgentDisambiguatorService.
"""

from backend.services.agent_service import AgentDisambiguatorService


def test_agent_disambiguator_empty_input():
    agent = AgentDisambiguatorService(api_key="mock_key")
    res = agent.polish_sentence("")
    assert res == ""


def test_agent_disambiguator_fallback_map():
    agent = AgentDisambiguatorService(api_key="")  # Force fallback
    res = agent.polish_sentence("mujhe paani chahiye", language="hi")
    assert res == "Mujhe paani chahiye."

    res_en = agent.polish_sentence("help me", language="en")
    assert res_en == "Please help me!"


def test_agent_disambiguator_unmapped_phrase_capitalization():
    agent = AgentDisambiguatorService(api_key="")
    res = agent.polish_sentence("random unmapped phrase")
    assert res == "Random unmapped phrase."
