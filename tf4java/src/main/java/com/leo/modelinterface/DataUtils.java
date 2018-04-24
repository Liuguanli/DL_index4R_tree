package com.leo.modelinterface;

import java.util.List;

public class DataUtils {

    public static void normalizeData(List<Float> latis, List<Float> longis, int featureRange) {
        if (latis == null || latis.size() == 0 || longis == null || longis.size() == 0 || latis.size() != longis.size() || featureRange <= 0)
            return;

        float maxLati = Float.MIN_VALUE;
        float maxLongi = Float.MIN_VALUE;
        float minLati = Float.MAX_VALUE;
        float minLongi = Float.MAX_VALUE;

        for (int i = 0; i < latis.size(); i++) {
            float lati = latis.get(i);
            float longi = longis.get(i);
            if (lati > maxLati) {
                maxLati = lati;
            }
            if (lati < minLati) {
                minLati = lati;
            }
            if (longi > maxLongi) {
                maxLongi = longi;
            }
            if (longi < minLongi) {
                minLongi = longi;
            }
        }
        float latiInterval = maxLati - minLati;
        float longiInterval = maxLongi - minLongi;
        for (int i = 0; i < latis.size(); i++) {
            latis.set(i, (latis.get(i) - minLati) * featureRange / latiInterval);
            longis.set(i, (longis.get(i) - minLongi) * featureRange / longiInterval);
        }
    }

}
