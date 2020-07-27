import cn.ict.zyq.bestConf.bestConf.sampler.DDSSampler;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.ProtectedProperties;

import java.util.ArrayList;
import java.util.Properties;

public class test {
    public static void main(String[] args){
        //模拟调优的第一个参数
        ArrayList<Attribute> atts = new ArrayList<Attribute>();
        Properties p1 = new Properties();
        p1.setProperty("range", "[1,6]");
        ProtectedProperties prop1 = new ProtectedProperties(p1);
        //模拟调优的第二个参数
        Properties p2 = new Properties();
        p2.setProperty("range", "[1,6]");
        ProtectedProperties prop2 = new ProtectedProperties(p2);


        ArrayList<String> attVals = new ArrayList<String>();
        for (int i = 0; i < 5; i++)
            attVals.add("val" + (i+1));
        atts.add(new Attribute("att1", prop1));
        atts.add(new Attribute("att2", prop2));
        //atts = {ArrayList@503}  size = 2
        // 0 = {Attribute@519} "@attribute att1 numeric"
        //  m_Name = "att1"
        //  m_Type = 0
        //  m_AttributeInfo = null
        //  m_Index = -1
        //  m_Weight = 1.0
        //  m_AttributeMetaInfo = {AttributeMetaInfo@526}
        //   m_Metadata = {ProtectedProperties@505}  size = 1
        //   m_Ordering = 1
        //   m_IsRegular = true
        //   m_IsAveragable = true
        //   m_HasZeropoint = true
        //   m_LowerBound = 1.0
        //   m_LowerBoundIsOpen = false
        //   m_UpperBound = 6.0
        //   m_UpperBoundIsOpen = false
        // 1 = {Attribute@520} "@attribute att2 numeric"

        //实例化
        DDSSampler sampler = new DDSSampler(3);

        //第一列轮数
        sampler.setCurrentRound(0);
        Instances data = sampler.sampleMultiDimContinuous(atts, 4, false);
        System.out.println(data);


        //sampler.setCurrentRound(1);
        //data = sampler.sampleMultiDimContinuous(atts, 2, false);
        //System.out.println(data);
        //
        //sampler.setCurrentRound(2);
        //data = sampler.sampleMultiDimContinuous(atts, 2, false);
        //System.out.println(data);
    }
}
