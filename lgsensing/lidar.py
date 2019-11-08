from rplidar import rplidar
from tcpcom import tcpcom

class Lidar:
    """
    This is the lidar class, based on RPLidar A1.
    """
    def __init__(self, port):
        self.port = port
        self.lidar = rplidar.RPLidar(self.port)
        self.reset()
        self.iterator = self.lidar.iter_scans()
        self.get_basic_info()

    def reset(self):
        self.lidar.stop()
        self.lidar.stop_motor()
        self.lidar.start_motor()

    def stop_lidar(self):
        self.lidar.stop()
        self.lidar.stop_motor()

    def get_basic_info(self):
        """ print info and health of lidar """
        print('--- Instantiated lidar device ---')
        info = self.lidar.get_info()
        health = self.lidar.get_health()
        print('info: {0}\nhealth: {1}'.format(info, health))

    def print_data(self):
        """
        Print lidar measurements continuously.

        :return: 3-tuple of (quality, angle, distance) where quality : int is
        reflected laser pulse strength, angle : float is measurement heading
        angle in degree unit [0, 360], distance : float is measured object
        distance from device's rotation centre. In millimeter unit.
        """
        while True:
            data = next(self.iterator)
            print(data)

    def query(self, angle):
        """ return the distance from lidar given a particular angle """
        data = next(self.iterator)
        nearest_angle = min([datum[1] for datum in data],
                            key=lambda d:abs(d-angle))
        datum = (datum for datum in data if datum[1] == nearest_angle)
        return next(datum)

    def serve_lidar_data(self, SERVER_PORT=22000):
        """ serve lidar data to remote machine in a given timeout"""

        def onStateChanged(state, msg):
            print('Server state: {0}\nMessage: {1}'.format(state, msg))
            while state == tcpcom.TCPServer.CONNECTED:
                data = next(self.iterator)
                print('Data sent: {}'.format(data))
                SERVER.sendMessage(str(data))

        print('--- Serving lidar data ---')
        SERVER = tcpcom.TCPServer(SERVER_PORT, stateChanged=onStateChanged)


