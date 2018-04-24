package com.leo.modelinterface;

import org.tensorflow.*;
import org.apache.commons.io.FileUtils;

import java.io.*;
import java.nio.FloatBuffer;
import java.util.*;

import static com.leo.modelinterface.DataUtils.normalizeData;


public class FullConnectedClassify {


    static Properties prop = new Properties();
    /**
     * 这里必须使用绝对路径
     * UnsatisfiedLinkError  if either the filename is not an
     *             absolute path name, the native library is not statically
     *             linked with the VM, or the library cannot be mapped to
     *             a native library image by the host system.
     */
    static {

        try {
            prop.load(new FileInputStream("config.properties"));
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.load(prop.getProperty("jni_path"));
    }

    public static void main(String args[]) {
        String testFilePath = prop.getProperty("test_files_path");
        FullConnectedClassify classify = new FullConnectedClassify();
        File dir = new File(testFilePath);
        long start = System.currentTimeMillis();
        if (dir.isDirectory()) {
            File[] files = dir.listFiles();
            for (int i = 0; i < files.length; i++) {
                File item = files[i];
                classify.execute(item.getAbsolutePath());
            }
        }
        long end = System.currentTimeMillis();
        System.out.println(end - start);
    }

    public void predict(List<Float> latis, List<Float> longis, List<Integer> indexes) {
        String modelPath = prop.getProperty("model_path");
        try (Graph graph = new Graph()) {
            byte[] bytes = FileUtils.readFileToByteArray(new File(modelPath));
            graph.importGraphDef(bytes);
            TensorFlowInferenceInterface tfi = new TensorFlowInferenceInterface(graph);
            Operation operation = tfi.graphOperation("layer3/prediction");
            Output output = operation.output(0);
            Shape shape = output.shape();
            int numClasses = (int) shape.size(1);

            for (int j = 0; j < latis.size(); j++) {
                FloatBuffer buffer = FloatBuffer.wrap(new float[]{latis.get(j), longis.get(j)});
                tfi.feed("x-input", buffer, 1, 2);
                tfi.run(new String[]{"layer3/prediction"}, false);//输出张量
                float[] outPuts = new float[numClasses];//结果分类
                tfi.fetch("layer3/prediction", outPuts);//接收结果 outPuts保存的即为预测结果对应的概率，最大的一个通常为本次预测结果}
                float max = Float.MIN_VALUE;
                int maxIndex = 0;
                for (int i = 0; i < outPuts.length; i++) {
                    if (outPuts[i] > max) {
                        max = outPuts[i];
                        maxIndex = i;
                    }
                }
//                System.out.println(" maxIndex:" + maxIndex + " realValue:" + indexes.get(j));
            }
            tfi.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    //
    public void handleResult(float lati, float longi, int predictedIndex, int realIndex) {
        // 预测正确
       if (predictedIndex == realIndex) {

       } else{
           // 使用还原的值去 Node里面查找
           /**
            * 1 先看是否 大于等于最大值99，或者小于等于最小值0
            * 如果是 则只需要查看一半
            *
            * 2 如果坐标明显不在MBR里面 那也不需要再去查看 那个node
            */
       }

    }

    public void execute(String fileName) {
        File file = new File(fileName);
        List<Float> latis = new ArrayList();
        List<Float> longis = new ArrayList();
        List<Integer> indexes = new ArrayList();
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String tempString = null;
            while ((tempString = reader.readLine()) != null) {

                String temps[] = tempString.split(",");
                latis.add(Float.valueOf(temps[0]));
                longis.add(Float.valueOf(temps[1]));
                indexes.add(Integer.valueOf(temps[2]));
            }
        } catch (IOException e) {

        }
        normalizeData(latis, longis, 20);
        predict(latis, longis, indexes);
    }


}
