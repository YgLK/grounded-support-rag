from support_graph.runtime.prompts import resolve_prompt_set


def test_v2_prompts_are_domain_neutral() -> None:
    prompts = resolve_prompt_set("v2")
    rendered = "\n".join(
        [
            str(prompts.route_query),
            str(prompts.evidence_grade),
            str(prompts.answer),
            str(prompts.non_answer),
        ]
    )

    assert "Kubernetes" not in rendered
    assert "deployments, rules, or fees" not in rendered
