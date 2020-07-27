package debugtools;

import java.util.ArrayList;

public class ReadingDebugTools {
    public static void getFunName(){
        StackTraceElement traceElement = ((new Exception()).getStackTrace())[1];
        System.out.println(traceElement.getMethodName()+"();");
    }

    /***
     * 二维数组显示为矩阵
     * @param lists
     */
    public static void arrayListAsMatrix(ArrayList<Integer>[]  lists){
        System.out.println("-----------------");
        for(ArrayList lst:lists){

            StringBuilder str = new StringBuilder();
            arrayListAsMatrix(lst);
            //str.append("[");
            //for (int i=0;i<lst.size();i++){
            //    str.append(lst.get(i)).append(",");
            //}
            //System.out.println(str.toString().substring(0, str.length() - 1)+"]");
            //System.out.println(",".trim(str.toString()));
        }
        System.out.println("-----------------");
    }


    /***
     *   一维数组显示为矩阵
     * @param lists
     */
    public static void arrayListAsMatrix(ArrayList<Integer>  lists){

            StringBuilder str = new StringBuilder();
            str.append("[");

            for (int i=0;i<lists.size();i++){
                str.append(lists.get(i)).append(",");
            }
            System.out.println(str.toString().substring(0, str.length() - 1)+"]");
    }

    public static void arrayListAsMatrix(int[] sampleA) {
        StringBuilder str = new StringBuilder();
        str.append("[");

        for (int i=0;i<sampleA.length;i++){
            str.append(sampleA[i]).append(",");
        }
        System.out.println(str.toString().substring(0, str.length() - 1)+"]");
    }

    public static void arrayListAsMatrix(ArrayList<Integer>[][] sets) {
        for (ArrayList[] l:sets){
            arrayListAsMatrix(l);
        }
    }
}
