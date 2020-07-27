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
package cn.ict.zyq.bestConf.cluster.InterfaceImpl;

import cn.ict.zyq.bestConf.cluster.Interface.Performance;

public class SUTPerformance implements Performance {

	private double throughPut;
	private double latency;
	
	public SUTPerformance(){
		
	}
	@Override
	public void initial(double throughPut, double latency) {
		this.throughPut = throughPut;
		this.latency = latency;
	}
	
	@Override
	public double getPerformanceOfThroughput() {
		return throughPut;
	}
	@Override
	public double getPerformanceOfLatency() {
		return latency;
	}
}
