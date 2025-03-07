import requests


class RobotControl():
    def __init__(self, url=None):
        super().__init__()

        self.prev_label = 0

        self.url = 'http://100.127.12.81:5100/robot'
        if not url == None:
            self.url = url

        self.sm_url = 'http://100.127.12.81:5100/statemachine'

        self.movements = ['left', 'right', 'stop']  # mapped to labels 0,1,2

    def command_robot(self, label):
        print(self.movements[label])
        objs = {}
        if not label == self.prev_label:
            if label == 0:
                objs = {'state': 'stop', 'direction': 'none'}
            else:
                objs = {'state': 'move', 'direction': self.movements[label]}
            print(objs)
            try:
                r = requests.post(self.url, params=objs, timeout=2)
                print(r)
            except Exception as e:
                print('Robot Webserver could not be reached!\n', e)
        self.prev_label = label

    def command_sm(self, label):
        print(label)
        objs = {'state': label}
        if label != 0:
            print(label)
            try:
                r = requests.post(self.sm_url, params=objs, timeout=1)
                print(r)
            except Exception as e:
                print('Robot Webserver could not be reached!\n', e)
