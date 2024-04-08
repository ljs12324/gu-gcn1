#!/usr/bin/env python
# encoding: utf-8

import logging
import os
import time
import random
import request
import subprocess
import thread
import numpy as np
from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel
from mininet.link import TCLink
from mininet.topo import Topo
from mininet.util import dumpNodeConnections
from mininet.log import output, info, error, warn, debug
import string
from scapy.layers.inet import *
from scapy.all import *


logger = logging.getLogger(__name__)


bw_a2c = 100
bw_a2e = 100
bw_h2e = 100

REQUEST_NUMBER = 30000
ATTACK_NUMBER = 100000

class Fattree(Topo):
    logger.debug("Class Fattree")
    CoreSwitchList = []
    AggSwitchList = []
    EdgeSwitchList = []
    HostList = []

    def __init__(self, k, density):
        logger.debug("Class Fattree init")
        self.pod = k
        self.iCoreLayerSwitch = (k / 2) ** 2
        self.iAggLayerSwitch = k * k / 2
        self.iEdgeLayerSwitch = k * k / 2
        self.density = density
        self.iHost = self.iEdgeLayerSwitch * density

        # Init Topo
        Topo.__init__(self)

    def create_topo(self):
        self.create_core_switch(self.iCoreLayerSwitch)
        self.create_agg_switch(self.iAggLayerSwitch)
        self.create_edge_switch(self.iEdgeLayerSwitch)
        self.create_host(self.iHost)

    def _add_switch(self, number, level, switch_list):
        swparam = {'protocols': 'OpenFlow13'}

        for x in xrange(1, number + 1):
            prefix = str(level) + "00"
            if x >= int(10):
                prefix = str(level) + "0"
            switch_list.append(self.addSwitch('s' + prefix + str(x), **swparam))

    def create_core_switch(self, number):
        logger.debug("Create Core Layer")
        self._add_switch(number, 1, self.CoreSwitchList)

    def create_agg_switch(self, number):
        logger.debug("Create Agg Layer")
        self._add_switch(number, 2, self.AggSwitchList)

    def create_edge_switch(self, number):
        logger.debug("Create Edge Layer")
        self._add_switch(number, 3, self.EdgeSwitchList)

    def create_host(self, number):
        logger.debug("Create Host")
        for x in xrange(1, number + 1):
            prefix = "h00"
            if x >= int(10):
                prefix = "h0"
            elif x >= int(100):
                prefix = "h"
            self.HostList.append(self.addHost(prefix + str(x)))

    def create_link(self, bw_h2e=0.2):

        logger.debug("Add link Core to Agg.")
        end = self.pod / 2
        for x in xrange(0, self.iAggLayerSwitch, end):
            for i in xrange(0, end):
                for j in xrange(0, end):
                    a = x + i
                    c = i * end + j
                    m = "bw_200" + str(a + 1) + "-100" + str(c + 1)
                    self.addLink(
                        self.CoreSwitchList[i * end + j],
                        self.AggSwitchList[x + i],bw=bw_a2c)
                        # bw=bw_a2c[m])

        logger.debug("Add link Agg to Edge.")

        for x in xrange(0, self.iAggLayerSwitch, end):
            for i in xrange(0, end):
                for j in xrange(0, end):
                    e = x + j
                    a = x + i
                    m = "bw_200" + str(a + 1) + "-300" + str(e + 1)
                    self.addLink(
                        self.AggSwitchList[x + i], self.EdgeSwitchList[x + j],bw=bw_a2e)
                        #bw=bw_a2e[m])

        logger.debug("Add link Edge to Host.")
        for x in xrange(0, self.iEdgeLayerSwitch):
            for i in xrange(0, self.density):
                self.addLink(
                    self.EdgeSwitchList[x],
                    self.HostList[self.density * x + i],
                    bw=bw_h2e)

    def set_ovs_protocol_13(self):
        self._set_ovs_protocol_13(self.CoreSwitchList)
        self._set_ovs_protocol_13(self.AggSwitchList)
        self._set_ovs_protocol_13(self.EdgeSwitchList)

    @staticmethod
    def _set_ovs_protocol_13(sw_list):
        for sw in sw_list:
            cmd = "sudo ovs-vsctl set bridge %s protocols=OpenFlow13" % sw
            os.system(cmd)

def ping_test(net):
    logger.debug("Start Test all network")
    net.pingAll()


def create_topo():
    logging.debug("LV1 Create Fattree")
    # 3 layer; 4 core switch; host density is 2.
    # topo = Fattree(4, 2)
    #6 fork fat tree, 72 hosts
    # topo = Fattree(6, 4)
    topo = Fattree(3, 4)
    topo.create_topo()
    # bandwidth unit Mbits
    topo.create_link(bw_h2e)

    logging.debug("LV1 Start Mininet")
    controller_ip = "127.0.0.1"
    controller_port = 6633
    net = Mininet(topo=topo, link=TCLink, controller=None, autoSetMacs=True)
    net.addController('controller', controller=RemoteController,
                      ip=controller_ip, port=controller_port)
    net.start()
    time.sleep(10)
    net.pingAll()

    CLI(net)
    net.stop()


if __name__ == '__main__':
    setLogLevel('info')
    if os.getuid() != 0:
        logger.debug("You are NOT root")
    elif os.getuid() == 0:
        create_topo()
