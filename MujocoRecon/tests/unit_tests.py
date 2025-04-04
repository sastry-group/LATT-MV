import unittest
from learning_table_tennis_from_scratch import hysr_one_ball

import context


class _ContactInfo:
    def __init__(self):
        self.minimal_distance = float("+inf")
        self.contact_occured = False


class HysrTest(unittest.TestCase):
    def test_reward_smash(self):
        class _Conf:
            def __init__(self):
                self.smash = None
                self.min_d_racket = None
                self.min_d_target = None
                self.max_b_vel = None
                self.c = None
                self.rtt_cap = None

            def reward(self):
                return hysr_one_ball._reward(
                    self.smash,
                    self.min_d_racket,
                    self.min_d_target,
                    self.max_b_vel,
                    self.c,
                    self.rtt_cap,
                )

        conf = _Conf()
        conf.smash = False
        conf.min_d_racket = float("+inf")
        conf.min_d_target = float("+inf")
        conf.max_b_vel = None
        conf.c = 1.0
        conf.rtt_cap = -0.2

        # reward 1 to 3 : did not touch the racket
        # but closer and closer to it
        r = []
        for d in (1.5, 1.0, 0.5):
            conf.min_d_racket = 1.5
            r.append(conf.reward())
        sorted_r = sorted(r)
        self.assertTrue(r == sorted_r)

        # reward 4:  touched the racket,
        # fixed velocity
        for smash in (True, False):
            r = []
            conf.smash = smash
            conf.min_d_racket = None
            conf.max_b_vel = 1
            for d in (1.5, 1.0, 0.5):
                conf.min_d_target = 1.5
                r.append(conf.reward())
            sorted_r = sorted(r)
            self.assertTrue(r == sorted_r)

    def test_ball_status_min_z_max_y(self):
        ball_status = context.BallStatus([0, 0, 0])
        ci = _ContactInfo()

        for pos in range(-2, 2):
            ball_status.update([pos, pos, pos], [1, 1, 1], ci)

        self.assertTrue(ball_status.min_z == -2)
        self.assertTrue(ball_status.max_y == +1)

    def test_ball_status_min_position_ball_target(self):
        ball_status = context.BallStatus([0, 0, 0])
        ci = _ContactInfo()
        ci.contact_occured = True

        ball_status.update([-2, -2, 2], [1, 1, 1], ci)
        ball_status.update([-0.1, -0.1, -0.1], [1, 1, 1], ci)
        ball_status.update([+0.5, +0.5, +0.5], [1, 1, 1], ci)

        self.assertTrue(
            all([ball_status.min_position_ball_target[i] == -0.1 for i in range(3)])
        )

    def test_ball_status_min_distance_ball_target(self):
        ball_status = context.BallStatus([0, 0, 0])
        ci = _ContactInfo()
        ci.contact_occured = True

        ball_status.update([-2, -2, 2], [1, 1, 1], ci)
        d1 = ball_status.min_distance_ball_target
        ball_status.update([-0.1, -0.1, -0.1], [1, 1, 1], ci)
        d2 = ball_status.min_distance_ball_target
        ball_status.update([+0.5, +0.5, +0.5], [1, 1, 1], ci)
        d3 = ball_status.min_distance_ball_target

        self.assertTrue(d2 < d1)
        self.assertTrue(d2 == d3)

    def test_ball_status_contact_occured(self):
        ball_status = context.BallStatus([0, 0, 0])
        ci = _ContactInfo()
        ci.contact_occured = True
        ball_status.update([-2, -2, 2], [1, 1, 1], ci)
        d1 = ball_status.min_distance_ball_racket
        self.assertTrue(d1 is None)

    def test_action_formatting(self):
        p_in = [1, 2, 1, 3, 1, 4]
        p_out = [(1, 2), (1, 3), (1, 4)]
        p = hysr_one_ball._convert_pressures_in(p_in)
        self.assertTrue(p == p_out)

        p_ago = [1, 2, 3]
        p_antago = [4, 5, 6]
        p = hysr_one_ball._convert_pressures_out(p_ago, p_antago)
        self.assertTrue(p == [1, 4, 2, 5, 3, 6])


if __name__ == "__main__":
    unittest.main()
