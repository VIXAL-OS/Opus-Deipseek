"""No-network checks for the standalone evaluation runner."""
import copy
import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import eval_firerouter as ev


class Block(NS):
    def model_dump(self):
        return vars(self)


def response(tool=False, reason=None):
    blocks = [Block(type='thinking', thinking='', signature='signed')]
    blocks.append(Block(type='tool_use', id='search-1', name='web_search', input={'query': 'release'})
                  if tool else Block(type='text', text='Answer'))
    return NS(content=blocks, model='claude-opus-5-5',
              stop_reason=reason or ('tool_use' if tool else 'end_turn'),
              usage=NS(model_dump=lambda: {'input_tokens': 100, 'output_tokens': 50}))


class Pilot(unittest.TestCase):
    def test_dry_run_does_not_read_keys_or_load_sdk(self):
        with patch('sys.argv', ['eval_firerouter.py']), patch.dict('os.environ', {}, clear=True), redirect_stdout(io.StringIO()) as output:
            ev.main()
        self.assertIn('DRY RUN', output.getvalue())

    def test_tool_history_and_usage_survive_across_followups(self):
        snapshots = []
        responses = iter([response(True), response(), response()])
        def create(**kwargs):
            snapshots.append(copy.deepcopy(kwargs))
            return next(responses)
        case = dict(system='Original system', turns=['Search', 'Follow up'], tool=True)
        result = ev.run_session(NS(messages=NS(create=create)), 'firerouter/opus', case, 2048)
        self.assertEqual([len(r['messages']) for r in snapshots], [1, 3, 5])
        self.assertEqual(snapshots[-1]['messages'][1]['content'][0].signature, 'signed')
        self.assertTrue(all(r['system'] == case['system'] for r in snapshots))
        self.assertEqual(len(result['calls']), 3)
        self.assertTrue(all(turn['complete'] for turn in result['turns']))

    def test_error_retains_earlier_billed_call_and_hides_error_body(self):
        client = NS(messages=NS(create=Mock(side_effect=[response(True), RuntimeError('private error body') ])))
        result = ev.run_session(client, 'firerouter/opus', dict(system='S', turns=['Q'], tool=True), 2048)
        self.assertEqual(len(result['calls']), 1)
        self.assertEqual(result['error_type'], 'RuntimeError')
        self.assertFalse(result['turns'][0]['complete'])
        self.assertNotIn('private error body', str(result))

    def test_tool_limit_preserves_all_calls_and_truncation_is_incomplete(self):
        client = NS(messages=NS(create=Mock(side_effect=[response(True) for _ in range(4)])))
        case = dict(system='S', turns=['Q'], tool=True)
        result = ev.run_session(client, 'firerouter/opus', case, 2048)
        self.assertEqual(result['error_type'], 'ToolRoundLimit')
        self.assertEqual(len(result['calls']), 4)
        client.messages.create = Mock(return_value=response(reason='max_tokens'))
        result = ev.run_session(client, 'firerouter/opus', case, 2048)
        self.assertFalse(result['turns'][0]['complete'])


if __name__ == '__main__':
    unittest.main()
