# 课程作业之BestConfig源码解析

# 参考
获取函数方法名
ref: https://www.cnblogs.com/likwo/archive/2012/06/16/2551672.html
``` 
package debugtools;
public class ReadingDebugTools {
    public static void getFunName(){
        StackTraceElement traceElement = ((new Exception()).getStackTrace())[1];
        System.out.println("funname: "+traceElement.getMethodName()+"();");
    }
}

使用：
ReadingDebugTools.getFunName();

```


