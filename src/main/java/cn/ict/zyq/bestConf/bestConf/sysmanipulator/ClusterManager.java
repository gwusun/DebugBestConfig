/**
 * Copyright (c) 2017 Institute of Computing Technology, Chinese Academy of Sciences, 2017 
 * Institute of Computing Technology, Chinese Academy of Sciences contributors. All rights reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License"); you
 * may not use this file except in compliance with the License. You
 * may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied. See the License for the specific language governing
 * permissions and limitations under the License. See accompanying
 * LICENSE file.
 */
package cn.ict.zyq.bestConf.bestConf.sysmanipulator;

import weka.core.Attribute;
import weka.core.Instances;

import java.util.Map;

public interface ClusterManager {
	// 关闭机器
	public void shutdown();

	public void test(int timeToTest);
	
	//only run those instance without the performance attribute
	//只运行那些没有性能属性的实例
	public Instances runExp(Instances samplePoints, String perfAttName);

	//输出日志到文件，为了调试方便
	public double setOptimal(Map<Attribute, Double> attributeToVal);
	
	/**collect the performances for part of samplePoints*/
	//收集部分采样点的性能
	public Instances collectPerfs(Instances samplePoints, String perfAttName);
}
