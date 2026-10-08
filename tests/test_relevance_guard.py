from agent.relevance_guard import guard_query


def test_off_topic_question_is_blocked():
    result = guard_query("Who won the cricket match yesterday?")
    assert result["allowed"] is False
    assert result["reason"] == "OFF_TOPIC"
    assert result["suggestions"]


def test_case_question_is_allowed():
    result = guard_query("What evidence supports the defendant's claim?")
    assert result["allowed"] is True


def test_legal_crime_terms_are_not_blocked_as_off_topic():
    result = guard_query("What evidence supports the assault allegation?")
    assert result["allowed"] is True


def test_empty_question_is_blocked():
    result = guard_query("")
    assert result["allowed"] is False
