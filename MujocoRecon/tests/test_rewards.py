import unittest
import tempfile
from learning_table_tennis_from_scratch.rewards import JsonReward, Reward, SmashReward


class REWARDS_TESTCASE(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def _get_json_file(self, normalization_constant, rtt_cap, smash):
        f = tempfile.NamedTemporaryFile(mode="w")
        fields = ",\n    ".join(
            ['"normalization_constant":{}', '"rtt_cap":{}', '"smash":{}']
        ).format(normalization_constant, rtt_cap, smash)
        content = "{{\n    {f}\n}}".format(f=fields)
        f.write(content)
        f.flush()
        return f

    def test_read_default_json(self):
        """
        testing reading the default config file
        does not thrown an exception
        """
        jsonpath = JsonReward.default_path()
        no_exception = True
        try:
            JsonReward.get(jsonpath)
        except Exception:
            no_exception = False
        self.assertTrue(no_exception)

    def test_json_parsing(self):
        """
        testing a json file results in correct
        config, including the instantion of
        a Reward
        """
        f = self._get_json_file(0.2, -0.3, "false")
        reward_fn = JsonReward.get(f.name)
        f.close()
        self.assertEqual(reward_fn.__class__, Reward)
        self.assertEqual(reward_fn.config.normalization_constant, 0.2)
        self.assertEqual(reward_fn.config.rtt_cap, -0.3)

    def test_json_parsing_smash(self):
        """
        testing a json file results in correct
        config, including the instantion of
        a SmashReward
        """
        f = self._get_json_file(1, 2, "true")
        reward_fn = JsonReward.get(f.name)
        f.close()
        self.assertEqual(reward_fn.__class__, SmashReward)
        self.assertEqual(reward_fn.config.normalization_constant, 1.0)
        self.assertEqual(reward_fn.config.rtt_cap, 2.0)

    def test_reward_values(self):
        """
        testing not the exact reward value,
        but values relative one to the other
        (e.g. reward of a ball flying far from
        the racket should be lower than the reward
        of a ball flying close to the racket)
        """
        for swing in (True, False):
            if swing:
                f = self._get_json_file(3.0, -0.2, "true")
            else:
                f = self._get_json_file(3.0, -0.2, "false")

            reward_fn = JsonReward.get(f.name)

            class Case:
                def __init__(
                    self,
                    min_distance_ball_racket,
                    min_distance_ball_target,
                    max_ball_velocity,
                ):
                    self.min_distance_ball_racket = min_distance_ball_racket
                    self.min_distance_ball_target = min_distance_ball_target
                    self.max_ball_velocity = max_ball_velocity

                def get(self, reward_fn):
                    return reward_fn(
                        self.min_distance_ball_racket,
                        self.min_distance_ball_target,
                        self.max_ball_velocity,
                    )

            # cases, from lower to higher expected rewards
            cases = []
            # did not touch the racket, far from racket
            cases.append(Case(2, None, None).get(reward_fn))
            # did not touch the racket, close to racket
            cases.append(Case(0.5, None, None).get(reward_fn))
            # touched the racket, far from target, slow
            cases.append(Case(None, 2, 1).get(reward_fn))
            # touched the racket, far from target, fast
            cases.append(Case(None, 2, 2).get(reward_fn))
            # checking all good
            for case1, case2 in zip(cases, cases[1:]):
                self.assertTrue(case1 <= case2)
            # cases, from lower to higher expected rewards
            case_far = Case(None, 2, 1).get(reward_fn)
            case_close = Case(None, 1, 1).get(reward_fn)
            self.assertTrue(case_far < case_close)
