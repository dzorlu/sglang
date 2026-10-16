import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _tools():
    return [
        Tool(
            type="function",
            function=Function(
                name="get_tool_schemas",
                description="load tool definitions by name",
                parameters={
                    "type": "object",
                    "properties": {
                        "names": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["names"],
                },
            ),
        )
    ]


class TestGlm47FullAssistantConstraintKnob(unittest.TestCase):
    """A client that declares a few tools and resolves the rest itself (a tool
    index: the model loads definitions by name, then calls the loaded tool)
    cannot work under the always-on glm47 grammar, which only admits the
    declared names. The knob turns the grammar off; the parser still runs."""

    def test_grammar_is_on_by_default(self):
        with envs.SGLANG_GLM47_DISABLE_FULL_ASSISTANT_CONSTRAINT.override(False):
            parser = FunctionCallParser(_tools(), "glm47")
            self.assertTrue(parser.detector.parses_required_natively())
            constraint = parser.get_structure_constraint(
                "auto", parallel_tool_calls=False, thinking_mode=True
            )
        self.assertIsNotNone(constraint)
        self.assertEqual(constraint[0], "full_assistant_ebnf")
        self.assertIn("get_tool_schemas", constraint[1])

    def test_knob_removes_the_grammar_for_auto(self):
        with envs.SGLANG_GLM47_DISABLE_FULL_ASSISTANT_CONSTRAINT.override(True):
            parser = FunctionCallParser(_tools(), "glm47")
            self.assertFalse(parser.detector.parses_required_natively())
            constraint = parser.get_structure_constraint(
                "auto", parallel_tool_calls=False, thinking_mode=True
            )
        self.assertIsNone(constraint)

    def test_knob_keeps_parsing_and_forwards_unknown_tools_with_the_other_knob(self):
        text = (
            "<tool_call>apps__crm__query<arg_key>query</arg_key>"
            "<arg_value>SELECT Id FROM Account</arg_value></tool_call>"
        )
        with (
            envs.SGLANG_GLM47_DISABLE_FULL_ASSISTANT_CONSTRAINT.override(True),
            envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(True),
        ):
            parser = FunctionCallParser(_tools(), "glm47")
            normal_text, calls = parser.parse_non_stream(text)
        self.assertEqual(normal_text, "")
        self.assertEqual([call.name for call in calls], ["apps__crm__query"])
        self.assertEqual(calls[0].tool_index, -1)


if __name__ == "__main__":
    unittest.main()
