#!/bin/bash
 # 设置Jmeter的环境
JMETER_HOME=/root/apache-jmeter-3.1
path=$JMETER_HOME/testresults.txt

# 切换工作目录
cd $JMETER_HOME/bin

#运行负载
nohup ./jmeter -n -t test4bestconf.jmx > $path &
