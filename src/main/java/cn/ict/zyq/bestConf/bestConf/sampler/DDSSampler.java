/**
 * Copyright (c) 2017 Institute of Computing Technology, Chinese Academy of Sciences, 2017
 * Institute of Computing Technology, Chinese Academy of Sciences contributors. All rights reserved.
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License"); you
 * may not use this file except in compliance with the License. You
 * may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied. See the License for the specific language governing
 * permissions and limitations under the License. See accompanying
 * LICENSE file.
 */
package cn.ict.zyq.bestConf.bestConf.sampler;

import debugtools.ReadingDebugTools;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.ProtectedProperties;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Properties;
import java.util.Random;

public class DDSSampler extends ConfigSampler {

    private static Random uniRand = new Random(System.nanoTime());

    private int rounds = 1;

    public DDSSampler(int totalRounds) {
        rounds = totalRounds;
    }

    //判断现在的样本是否和原来的样本一样
    private static boolean inAlready(ArrayList<Integer>[][] sets, ArrayList<Integer>[] permNew) {
        //permNew 上一步产生的矩阵
        // [0,1,2]
        // [1,0,2]
        boolean noSame = true, notEqual = false;

        //sets.length 返回样本量的大小
        for (int i = 0; i < sets.length; i++) {
            if (sets[i] != null) {//compare the two sample
                notEqual = false;
                //compare from the second attribute
                for (int attR = 1; attR < permNew.length; attR++) {
                    //compare the permutation of this attribute
                    for (int p = 0; p < permNew[attR].size(); p++) {
                        if (sets[i][attR].get(p) != permNew[attR].get(p)) {
                            notEqual = true;
                            break;
                        }
                    }
                    if (notEqual)
                        break;//no need to compare other attributes
                }
                if (!notEqual) {
                    noSame = false;
                    break;
                }
            } else//no more recursion
                break;
        }
        return !noSame;
    }

    /***
     * 将第一个和最大的一个交交换，就是将上文中参数的 sets 的第一个换位 distance 最大的那个
     * @param sets
     * @param dists
     * @param pos1
     * @param pos2
     */
    private static void positionSwitch(ArrayList<Integer>[][] sets, long[] dists, int pos1, int pos2) {

        ArrayList<Integer>[] tempSet = sets[pos1];
        sets[pos1] = sets[pos2];
        sets[pos2] = tempSet;

        long tempVal = dists[pos1];
        dists[pos1] = dists[pos2];
        dists[pos2] = tempVal;
    }

    long dists[] = null;
    ArrayList<Integer>[][] sets = null;
    private int sampleSetToGet = 0;

    public void setCurrentRound(int crntRound) {
        if (sets != null && crntRound < sets.length)
            sampleSetToGet = crntRound;
        sampleSetToGet = 0;
    }

    public void resetRound() {
        sets = null;
        sampleSetToGet = 0;
        dists = null;

        File f = new File("data/000SAMPLING_RESET_" + System.currentTimeMillis());
        try {
            System.out.println("creating file " + f.createNewFile());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     *  抽样算法的实现
     * At current version, we assume all attributes are numeric attributes with bounds
     * 在当前版本中，我们假设所有属性都是带有边界的数值属性
     * @param useMid true if to use the middle point of a subdomain, false if to use a random point within a subdomain
     *               如果使用子域的中间点，则使用true;如果使用子域内的随机点，则使用false
     */
    public Instances sampleMultiDimContinuous(ArrayList<Attribute> atts, int sampleSetSize, boolean useMid) {
        ReadingDebugTools.getFunName();

        /* ==============================确定抽哪个格子（二维格子为例）==============================*/
        //计算的distance 最大的那个数组
        ArrayList<Integer>[] crntSetPerm;
        //only initialize once 只初始化一次
        if (sets == null) {
            //possible number of sample sets will not exceed $sampleSetSize to the power of 2
            // 样本数量：可能的样本集数量不会超过$sampleSetSize的2次幂，以后要拿这个来进行参数切分。
            //			轮数同样决定样本的数量
            int L = (int) Math.min(rounds,
                    atts.size() > 2 ? Math.pow(sampleSetSize, atts.size() - 1) :
                            (atts.size() > 1 ? sampleSetSize : 1));

            //initialization
            //距离 distances 向量，两点之间连线的距离，保存的是 generateOneSampleSet 产生的抽样矩阵中最小的值
            dists = new long[L];
            //样本集 sets 向量，l是其样本的数量
            sets = new ArrayList[L][];

            //初始化距离为 -1，样本集为null
            for (int i = 0; i < L; i++) {
                dists[i] = -1; // set -1 for all item in dists
                sets[i] = null; //tow dimension array
            }

            //算出来的 欧几里得距离中最大的值 和 索引
            long maxMinDist = -1;
            int posWithMaxMinDist = -1;

            //generate L sets of sampleSetSize points
            ////两个参数，三个样本结果
            //				[0,1,2]
            //				[1,0,2]
            //
            //两个参数，四个样本结果
            //				[0,1,2,4]
            //				[1,0,2,3]

            for (int i = 0; i < L; i++) {
                //attrs 属性存放的数组，参数的个数
                ArrayList<Integer>[] setPerm = generateOneSampleSet(sampleSetSize, atts.size());
                //两个参数，三个样本结果
                //[0,1,2]
                //[1,0,2]

                //两个参数，四个样本结果
                //[0,1,2,4]
                //[1,0,2,3]

                //由此可见，其生成的结果是以参数个数为行的数量，样本数量为列的数量。
                //第一行的值为1,2,...,样本数量
                //列二行的值为xi={random x,1<x样本数量}，且值唯一，二行的值为第一行的一个排列。

                //continue the samples set generation till different samples are obtained
                //继续样本集的生成，直到得到不同的样本
                while (inAlready(sets, setPerm))
                    setPerm = generateOneSampleSet(sampleSetSize, atts.size());
                sets[i] = setPerm;

				System.out.println("++++++++++++++生成的一个样本集 =========================="+i);
				ReadingDebugTools.arrayListAsMatrix(setPerm);
				/*
第一个：setPerm
[0,1,2,3]
[2,1,0,3]
第二个setPerm
[0,1,2,3]
[0,3,1,2]
第三个setPerm
[0,1,2,3]
[1,2,0,3]
				* */

                //compute the minimum distance minDist between any sample pair for each set
                //计算setPerm集合中任意样本对之间的最小距离。欧几里得距离，即两点之间的长度

                //sets [i] 的最小距离为 dists[i]
                dists[i] = minDistForSet(setPerm);

                //select the set with the maximum minDist
                //选择 dists 中距离值最大的
                if (dists[i] > maxMinDist) {
                    posWithMaxMinDist = i;
                    maxMinDist = dists[i];
                }
            }
            //now let the first sample set be the one with the max mindist
            //将第一个和最大的一个交交换，就是将上文中参数的 sets 的第一个换位 distance 最大的那个

            positionSwitch(sets, dists, 0, posWithMaxMinDist);
        }
        //计算的distance 最大的那个数组
        crntSetPerm = sets[sampleSetToGet];
        /* ==============================确定抽哪个格子（二维为例）==============================*/




        /* =========================根据参数范围、样本量划分区间===================================*/

        //generate and output the set with the maximum minDist as the result、
        //生成并输出minDist最大的集合作为结果

        //first, divide the domain of each attribute into sampleSetSize equal subdomain
        // 比如所有三个样本，两个参数
        // 每个参数都会被三次（三个样本，样本次数）划分，产生4个值,如  [1,6] ==> [1.0,2.666666666666667,4.333333333333334,6.0]
        //首先，将每个属性的域划分为sampleSetSize等子域
        double[][] bounds = new double[atts.size()][sampleSetSize + 1];//sampleSetSize+1 to include the lower and upper bounds
        Iterator<Attribute> itr = atts.iterator();
        Attribute crntAttr;
        boolean[] roundToInt = new boolean[atts.size()];

        for (int i = 0; i < bounds.length; i++) {
			/*
			这个是用户传递过来的那个参数范围，如[1，6]
			* m_Name = "att1"
				m_Type = 0
				m_AttributeInfo = null
				m_Index = -1
				m_Weight = 1.0
				m_AttributeMetaInfo = {AttributeMetaInfo@624}
				 m_Metadata = {ProtectedProperties@625}  size = 1
				 m_Ordering = 1
				 m_IsRegular = true
				 m_IsAveragable = true
				 m_HasZeropoint = true
				 m_LowerBound = 1.0
				 m_LowerBoundIsOpen = false
				 m_UpperBound = 6.0
				 m_UpperBoundIsOpen = false
			* */
            crntAttr = itr.next();

            // [1,6] ==> [1.0，2.25，3.5，4.75，6.0]
            uniBoundsGeneration(bounds[i], crntAttr, sampleSetSize);
            //flexibleBoundsGeneration(bounds[i], crntAttr, sampleSetSize);

            if (bounds[i][sampleSetSize] - bounds[i][0] > sampleSetSize)
                roundToInt[i] = true;
        }

        /* =========================根据参数范围、样本量划分区间===================================*/



        /* =========================在选好的格子中随机产生值===================================*/
        //
        //  bounds=[1.0 2.666666666666667 4.333333333333334 6.0] x的划分范围
        //         [1.0 2.666666666666667 4.333333333333334 6.0] y的划分范围

        //second, generate the set according to setWithMaxMinDist
        //其次，生成集合，规则是集合之间的最大距离
        Instances data = new Instances("SamplesByLHS", atts, sampleSetSize);
        for (int i = 0; i < sampleSetSize; i++) {
            double[] vals = new double[atts.size()];
            //每次确定绑定的值
            for (int j = 0; j < vals.length; j++) {
                // 在已划分的区间中，选定一个区间的分界点的之后，在这个点与下一个点的这个间隔内随机取一个点，
                // 比如[1,2,3,4] 是一个已划分的区间，
                // 根据上面的 crntSetPerm 确定选中的一个点为 2，那么他就在 [2,3]之间随机取一个点。即 2+(3-2)*uniRand.nextDouble()
                //
                vals[j] = useMid ?
                        (bounds[j][crntSetPerm[j].get(i)] + bounds[j][crntSetPerm[j].get(i) + 1]) / 2 :
                        bounds[j][crntSetPerm[j].get(i)] +
                                (
                                        (bounds[j][crntSetPerm[j].get(i) + 1] - bounds[j][crntSetPerm[j].get(i)]) * uniRand.nextDouble()
                                );
                // bounds[j][crntSetPerm[j].get(i)]+bounds[j][crntSetPerm[j].get(i)+1])/2 取中间的一个点
                //  bounds=[1.0 2.666666666666667 4.333333333333334 6.0] x的划分范围
                //         [1.0 2.666666666666667 4.333333333333334 6.0] y的划分范围
                //	vals[j]= bounds[j][crntSetPerm[j].get(i)]+ ((bounds[j][crntSetPerm[j].get(i)+1]-bounds[j][crntSetPerm[j].get(i)])*uniRand.nextDouble())
                //		 	  bounds[j][crntSetPerm[j].get(i)]：根据之前生成的抽样模板数组，获取值：
                //			 (bounds[j][crntSetPerm[j].get(i)+1]-bounds[j][crntSetPerm[j].get(i)]) 当前值到下一个值之间的间隔，乘以一个随机数，就是在这个区间随机取点
                //
                //		     当前绑定参数+抽样模板
                if (roundToInt[j])
                    vals[j] = (int) vals[j];
            }
            //给这个范围加个权重，没别的意思，只是这里的权重都为1而已
            data.add(new DenseInstance(1.0, vals));
        }
        //data=      "1,2"
        // 			 "3,5"
        // 			 "4,3"
        /* =========================在选好的格子中随机产生值===================================*/


        //third, return the generated points
        //第三，返回生成的点
        return data;
    }

    /**
     * 将区间等分，区间范围为用户给定的范围，步长为 (max-min)/sampleSetSize
     * [1,6] ==> [1.0，2.25，3.5，4.75，6.0]
     * @param bounds
     * @param crntAttr
     * @param sampleSetSize
     */
    private static void uniBoundsGeneration(double[] bounds, Attribute crntAttr, int sampleSetSize) {
        //第一个和最后一个设为用户指定的范围值，如 [1,6] 设置 bounds[0]=1,bounds[sampleSetSize]=6
        bounds[0] = crntAttr.getLowerNumericBound();
        bounds[sampleSetSize] = crntAttr.getUpperNumericBound();
        //获得距离，间隔距离
        double pace = (bounds[sampleSetSize] - bounds[0]) / sampleSetSize;
        for (int j = 1; j < sampleSetSize; j++) {
            bounds[j] = bounds[j - 1] + pace;
        }
    }

    /**
     * generate one sample set based on the requirement of LHS sampling method
     * 根据LHS抽样方法的要求生成一个样本集
     *
     * Latin-Hypercube Sampling:https://www.sciencedirect.com/topics/engineering/latin-hypercube-sampling
     * 分层次抽样方法
     * 在统计抽样中，拉丁方阵是指每行、每列仅包含一个样本的方阵
     *
     * @return the generated sample set that specifies which subdomain to choose under each attributed for each sample
     * 			each arraylist is a permutation of the subdomains for each attribute
     * 		    生成的示例集指定在每个示例的每个属性下选择哪个子域，每个arraylist是每个属性的子域的排列
     */
    private static ArrayList<Integer>[] generateOneSampleSet(int sampleSetSize, int attrNum) {

        ReadingDebugTools.getFunName();
        //attrNum 参数的个数，比如线程数量和内存大小
        ArrayList<Integer>[] setPerm = new ArrayList[attrNum];//sampleSetSize samples; each with atts.size() attributes
        int crntRand;
        //generate atts.size() permutations of sampleSetSize integers
        // 生成参数个数对sampleSetSize整数的排列
        //		start from the second attribute, the first attribute always uses the natural order
        for (int i = 1; i < attrNum; i++) {
            setPerm[i] = new ArrayList<Integer>(sampleSetSize);

            //randomly generate a permutation for sampleSetSize integers
            //：为sampleSetSize整数随机生成一个排列
            for (int j = 0; j < sampleSetSize; j++) {
                crntRand = uniRand.nextInt(sampleSetSize);

                //for each set, each subdomain of any parameter has one and only one sample in it
				// 保证唯一
                while (setPerm[i].contains(crntRand)) {
                    crntRand = uniRand.nextInt(sampleSetSize);
                }
                setPerm[i].add(crntRand);
            }
        }
        //the first attribute always uses the natural order
        setPerm[0] = new ArrayList<Integer>(sampleSetSize);
        for (int j = 0; j < sampleSetSize; j++) {
            setPerm[0].add(j);

        }
        return setPerm;
    }

    /**
     * compute the minimum distance between any sample pair in the set of setPerm
     * 计算setPerm集合中任意样本对之间的最小距离
     * 调试输出
     * 0,1,2,
     * 1,2,0,
     * 计算setPerm集合中任意样本对之间的最小距离 minDistForSet------------
     * 0,1,-
     * 	1,2,dist:2
     * 	2,0,dist:5
     * ====================
     * 1,2,-
     * 	2,0,dist:5
     * ====================
     *
     * 结论
     * 行===参数个数
     * 列===样本数量
     *
     * 0,1,2,
     * 1,2,0,
     *
     * 第一列的 0 和其他列计算距离，分别是 1 和 2
     * 		   1 				       2    0
     *
     * 第二列和第三例计算距离
     *
     *
     * 返回最小的距离
     */
    private static long minDistForSet(ArrayList<Integer>[] setPerm) {
        ReadingDebugTools.getFunName();
		/*
		* setPerm = {ArrayList[2]@504}
					 0 = {ArrayList@509}  size = 3
					  0 = {Integer@512} 0
					  1 = {Integer@513} 1
					  2 = {Integer@514} 2
					 1 = {ArrayList@510}  size = 3
					  0 = {Integer@512} 2
					  1 = {Integer@513} 1
					  2 = {Integer@514} 0
		* */
        ReadingDebugTools.arrayListAsMatrix(setPerm);
        System.out.println("计算setPerm集合中任意样本对之间的最小距离 minDistForSet------------");
        long mindist = Long.MAX_VALUE, dist;
        //人为设置的样本集大小
        int sampleSetSize = setPerm[0].size();
        int[] sampleA = new int[setPerm.length], sampleB = new int[setPerm.length];
        for (int i = 0; i < sampleSetSize - 1; i++) {
            for (int j = 0; j < sampleA.length; j++) {
                sampleA[j] = setPerm[j].get(i);

            }
            ReadingDebugTools.arrayListAsMatrix(sampleA);
            System.out.println("-");
            //enumerate all combinations
            //列举所有的组合 两个参数的所有组合
            for (int k = i + 1; k < sampleSetSize; k++) {

                for (int j = 0; j < sampleB.length; j++) {
                    sampleB[j] = setPerm[j].get(k);
                }
                System.out.printf("\t");
                ReadingDebugTools.arrayListAsMatrix(sampleB);
                //欧几里得距离，两点间的长度
                dist = eucDistForPairs(sampleA, sampleB);
                System.out.println("dist:" + dist);
                mindist = mindist > dist ? dist : mindist;
            }
            System.out.println("====================");
        }
        return mindist;
    }

    /**
     * compute the Euclidean distance between two points in a multi-dim integer space
     * 计算多暗淡整数空间中两点间的欧氏距离,两点间的长度
     * Euclidean distance：欧几里得距离，两点间的长度
     * http://rosalind.info/glossary/euclidean-distance/
     */
    private static long eucDistForPairs(int[] sampleA, int[] sampleB) {
        long dist = 0;
        for (int i = 0; i < sampleA.length; i++)
            dist += (sampleA[i] - sampleB[i]) * (sampleA[i] - sampleB[i]);
        return dist;
    }

    public static void main(String[] args) {
        //模拟调优的第一个参数
        ArrayList<Attribute> atts = new ArrayList<Attribute>();
        Properties p1 = new Properties();
        p1.setProperty("range", "[0,1]");
        ProtectedProperties prop1 = new ProtectedProperties(p1);
        //模拟调优的第二个参数
        Properties p2 = new Properties();
        p2.setProperty("range", "[321,1E9]");
        ProtectedProperties prop2 = new ProtectedProperties(p2);
        //模拟调优的第三个参数
        Properties p3 = new Properties();
        p3.setProperty("range", "[1,30]");
        ProtectedProperties prop3 = new ProtectedProperties(p3);

        ArrayList<String> attVals = new ArrayList<String>();
        for (int i = 0; i < 5; i++)
            attVals.add("val" + (i + 1));
        atts.add(new Attribute("att1", prop1));
        atts.add(new Attribute("att2", prop2));
        atts.add(new Attribute("att3", prop3));

        //实例化
        DDSSampler sampler = new DDSSampler(3);

//第一轮抽样
sampler.setCurrentRound(0);
Instances data = sampler.sampleMultiDimContinuous(atts, 2, false);
System.out.println(data);

//第二轮抽样
sampler.setCurrentRound(01);
data = sampler.sampleMultiDimContinuous(atts, 2, false);
System.out.println(data);
//第三轮抽样
sampler.setCurrentRound(2);
data = sampler.sampleMultiDimContinuous(atts, 2, false);
System.out.println(data);
    }

}

