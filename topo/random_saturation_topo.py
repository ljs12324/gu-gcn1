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

# BW of agg switch to core switch
# unit Mbits
# bw_a2c = {
#     'bw_2001-1001': 100,
#     'bw_2001-1002': 20,
#     'bw_2002-1003': 100,
#     'bw_2002-1004': 20,
#     'bw_2003-1001': 20,
#     'bw_2003-1002': 20,
#     'bw_2004-1003': 20,
#     'bw_2004-1004': 20,
#     'bw_2005-1001': 100,
#     'bw_2005-1002': 20,
#     'bw_2006-1003': 20,
#     'bw_2006-1004': 20,
#     'bw_2007-1001': 100,
#     'bw_2007-1002': 20,
#     'bw_2008-1003': 100,
#     'bw_2008-1004': 20,
# }
#
# # BW of agg switch to edge switch
# bw_a2e = {
#     'bw_2001-3001': 20,
#     'bw_2001-3002': 20,
#     'bw_2002-3001': 20,
#     'bw_2002-3002': 20,
#     'bw_2003-3003': 20,
#     'bw_2003-3004': 20,
#     'bw_2004-3003': 10,
#     'bw_2004-3004': 10,
#     'bw_2005-3005': 10,
#     'bw_2005-3006': 4,
#     'bw_2006-3005': 4,
#     'bw_2006-3006': 4,
#     'bw_2007-3007': 20,
#     'bw_2007-3008': 20,
#     'bw_2008-3007': 20,
#     'bw_2008-3008': 20,
# }

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

def generate_traffic_by_hping(net, topo):
    #time.sleep(10)
    #attacker = net.get(topo.UserList[0])
    #Select the topo.HostList[0] as the attacker, so move it from the host list in sending traffic
    User = [topo.HostList[i] for i in range(0, len(topo.HostList))]
    # User = []
    # for i in range(8, len(topo.HostList)):
    #     User.append(topo.HostList[i])
    #13 pairs of hosts, sending background traffic by 1V1
    send_user = random.sample(range(0, len(User) / 2), 5)
    recv_user = random.sample(range(len(User) / 2, len(User)), 5)

    # period_time = np.random.poisson(3, size=len(send_user))
    for i in range(REQUEST_NUMBER):
        # time.sleep(1)
        # k = random.randint(0, len(send_user) - 1)
        # T = random.randint(45, 55)
        # send_user = random.sample(range(0, len(User) / 2), 4)
        # recv_user = random.sample(range(len(User) / 2, len(User)), 4)
        # for t in range(0, T):
        byte_count = random.sample(range(1, 10), len(send_user))
        packet_count = random.sample(range(1, 100), len(send_user))
        times = np.random.poisson(10, size=len(send_user))
        period_time = random.sample(range(1, 10), len(send_user))
        for k in range(0, len(send_user)):
            for j in range(0, len(recv_user)):
                sender = net.get(User[send_user[k]])
                recver = net.get(User[recv_user[j]])
                # byte_index = random.randint(0, len(rate)-1)
                send_traffic_by_hping(sender, recver, byte_count[j - 1], packet_count[j - 1])
        k = random.randint(0, len(send_user) - 1)
        time.sleep(period_time[k])

    # logger.info("end send tcp")
    pass

#hping3 --interval u10000表示每秒10个包，u1000表示每秒100个包，所以u100000应该是表示每秒1个包
# def send_traffic_by_hping(sender,User,net,byte_count, packet_count, rate):
def send_traffic_by_hping(sender, recver, byte_count, packet_count):
    recver_ip = recver.IP()
    sender_ip = sender.IP()
    #sender and receiver are randomly selected, must compare them
    if recver_ip != sender_ip:
        #0529 50 60 65 55 35
        # sender.popen("nping --tcp --rate 10 --data-length " + str(byte_count) + " --count " + str(
        #     packet_count) + " " + recver_ip, shell=True)
        # time.sleep(0.1)
        # # time.sleep(0.5)
        # sender.popen("nping --udp --rate 10 --data-length " + str(byte_count) + " --count " + str(
        #     packet_count) + " " + recver_ip, shell=True)
        # time.sleep(0.1)
        # time.sleep(0.5)
        sender.popen("nping --icmp --rate 10 --data-length " + str(byte_count) + " --count " + str(
            packet_count) + " " + recver_ip, shell=True)


        # print 'normal'
        # print(sender_ip, recver_ip)
    pass


#攻击也是多个线程的发送lddos流量,time.sleep(1):每次攻击周期为1s index:攻击者index


def generate_ldos_attack_by_hping(net, topo,srcip, dstip, srcindex,pythonfun):
    # #time.sleep(1)
    # # print("start send attacks")
    # logger.info("start send attacks")
    # # data = open("/home/ran/Desktop/openrainbow/openrainbow/DWT/flow_feature14.txt", 'a+')
    # # data.write("wolaiguo")
    # # data.close()
    # attacker_index = index1
    # # victim = [user for user in topo.HostList if user != topo.HostList[0] and user != topo.HostList[2]]
    # #select 7 host as victim randomly
    # #victim_count = random.randint(9, len(victim)-1)
    # # victim = random.sample(victim, 9)
    # # victim_index_list = random.sample(range(0, len(victim)), victim_count)
    # # victim = [topo.HostList[i] for i in range(16, len(topo.HostList))]
    # # victim = ['10.0.0.2', '10.0.0.4', '10.0.0.6', '10.0.0.8', '10.0.0.10', '10.0.0.12', '10.0.0.14', '10.0.0.16',
    # #           '10.0.0.18', '10.0.0.20', '10.0.0.20', '10.0.0.24', '10.0.0.26', '10.0.0.28', '10.0.0.30', '10.0.0.32']
    # # victim = ['10.0.0.25', '10.0.0.26', '10.0.0.27', '10.0.0.28', '10.0.0.29', '10.0.0.30', '10.0.0.31', '10.0.0.32']
    # victim = ['10.0.0.13', '10.0.0.14', '10.0.0.15', '10.0.0.16']
    # #attack_server_index = 0
    # period_time = np.random.poisson(150, size=len(victim))
    # attacker = net.get(topo.HostList[attacker_index])
    # for j in range(ATTACK_NUMBER):
    #     time.sleep(1)
    #     for t in range(0, 5):
    #         for v in victim:
    #             send_ldos_by_hping(attacker, v, net)
    #     k = random.randint(0, len(period_time)-1)
    #     time.sleep(period_time[k])
    # print('sending end')
    # pass

    # while True:
    #     t = time.time()
    #     packet = []
    #     src_ip = 11
    #     for i in range(4):
    #         randNum = random.randint(1, 100)
    #         for j in range(randNum):
    #             randNum1 = random.randint(1, 10)
    #             payload = ''.join(random.sample(
    #                 string.ascii_letters + string.digits + 'zyxwvutsrqponmlkjihgfedcbafafakjhwqiuh2783hr3iuh3764834sdkbaskdsakjdnaskdjs1321sad6a5s1d5asd1a3s2d16as5d413as1da3s5f46aw54rwakjnd,masnfiuashrlqkwnrqnrlqjhoialskdw,asmndajsnlwrnq,mnwrqwljhskanhda',
    #                 randNum1))
    #             p = IP(src="10.0.0." + str(src_ip), dst="10.0.0." + str(dst_ip)) / ICMP() / payload
    #             packet.append(p)
    #         src_ip += 1
    #     send(packet, verbose=0)
    #     time.sleep(10)
    attacker = net.get(topo.HostList[srcindex])
    print(attacker)
    attacker.popen("python "+pythonfun+" "+str(srcip)+" "+str(dstip)+" 0")

def get_arp(net, topo, ip_start, mac_src, dstindex):
    attacker = net.get(topo.HostList[dstindex])
    print(attacker)
    for i in range(8):
        ip_start += 1
        attacker.popen("arp -s 10.0.0." + str(ip_start) + " " + mac_src)

def testping(net, topo,srcindex):
    attacker = net.get(topo.HostList[srcindex])
    print(attacker)
    attacker.popen("python 22222222.py 45 2 0")


#nping --rate表示每秒发送数据包数量，--count表示发送轮次数量，在达到相应轮之后，停止
def send_ldos_by_hping(attacker, victim, net):
    # victim_1 = victim[victim_index]
    # victim_1 = net.get(victim_1)
    # victim_1_ip = victim_1.IP()
    attacker_ip = attacker.IP()
    # if attacker_ip != victim:
    # print attacker_ip
    #now --rate 150
    attacker.popen("nping --tcp --rate 20 --data-length 900 --count 250 " + victim, shell=True)
    time.sleep(0.1)
    attacker.popen("nping --udp --rate 20 --data-length 900 --count 250 " + victim, shell=True)
    time.sleep(0.1)
    attacker.popen("nping --icmp --rate 20 --data-length 900 --count 250 " + victim, shell=True)
    #10% 4 30% 12 50% 28 70% 65 90% 250
    # print 'dos'
    # print(attacker_ip,victim)
    pass


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
    thread.start_new_thread(get_arp, (net, topo, 43, "00:00:00:00:00:01", 14))
    thread.start_new_thread(get_arp, (net, topo, 43, "00:00:00:00:00:01", 15))
    thread.start_new_thread(get_arp, (net, topo, 43, "00:00:00:00:00:01", 13))
    thread.start_new_thread(get_arp, (net, topo, 43, "00:00:00:00:00:01", 12))
    # ping_test(net)
    time.sleep(10)
    net.pingAll()
    # thread.start_new_thread(testping, (net, topo, 0))
    time.sleep(60)
    # 创建1个线程，分别发送背景流量

    thread.start_new_thread(generate_traffic_by_hping, (net, topo,))
    # thread.start_new_thread(generate_ldos_attack_by_hping, (net, topo, 88, 16, 4, "11111111.py"))
    # time.sleep(1)
    thread.start_new_thread(generate_ldos_attack_by_hping, (net, topo, 44,13,0,"11111111.py"))
    time.sleep(50)

    # # 创建4个线程，发送攻击流量

    # thread.start_new_thread(generate_ldos_attack_by_hping, (net, topo, 1))
    # thread.start_new_thread(generate_ldos_attack_by_hping, (net, topo, 2))
    # thread.start_new_thread(generate_ldos_attack_by_hping, (net, topo, 3))
    # ping_test(net)
    # use if `swparam = {'protocols': 'OpenFlow13'}` failed.
    # topo.set_ovs_protocol_13()

    #open_service(net, topo)
    # logger.debug("LV1 dumpNode")
    # dumpNodeConnections(net.hosts)
    # dumpNodeConnections(net.switches)

    # Net_link = dumpNetConnections(net)
    # print type(Net_link)
    # print Net_link
    # pingTest(net)
    # iperfTest(net, topo)
    CLI(net)
    net.stop()


if __name__ == '__main__':
    setLogLevel('info')
    if os.getuid() != 0:
        logger.debug("You are NOT root")
    elif os.getuid() == 0:
        create_topo()
