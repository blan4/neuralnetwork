package org.seniorsigan.ml.examples.digits;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;

public class ImageService {
    public void toImage(final double[] data) throws IOException {
        int[] pixels = new int[data.length];
        for (int i = 0; i < data.length; ++i) {
            pixels[i] = (int)(data[i] * 255);
        }
        BufferedImage img = new BufferedImage(28,28,BufferedImage.TYPE_BYTE_GRAY);
        WritableRaster raster = img.getRaster();
        raster.setPixels(0,0,28,28, pixels);
        File f = File.createTempFile("image", ".jpg");
        ImageIO.write(img, "jpg", f);
        Runtime.getRuntime().exec("open " + f.getAbsolutePath());
    }
}
