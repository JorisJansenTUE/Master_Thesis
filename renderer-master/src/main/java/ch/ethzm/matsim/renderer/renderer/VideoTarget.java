package ch.ethzm.matsim.renderer.renderer;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.IOException;

import org.matsim.core.utils.misc.Time;

import ch.ethzm.matsim.renderer.config.RenderConfig;
import io.humble.video.Codec;
import io.humble.video.Encoder;
import io.humble.video.MediaPacket;
import io.humble.video.MediaPicture;
import io.humble.video.Muxer;
import io.humble.video.MuxerFormat;
import io.humble.video.PixelFormat;
import io.humble.video.Rational;
import io.humble.video.awt.MediaPictureConverter;
import io.humble.video.awt.MediaPictureConverterFactory;

public class VideoTarget {
	private final RenderConfig renderConfig;

	public VideoTarget(RenderConfig renderConfig) {
		this.renderConfig = renderConfig;
	}

	public void run(Renderer renderer) {
		try {
			Muxer muxer = Muxer.make(renderConfig.outputPath, null, null);

			MuxerFormat format = muxer.getFormat();
			Codec codec = Codec.findEncodingCodec(format.getDefaultVideoCodecId());

			Encoder encoder = Encoder.make(codec);
			encoder.setWidth(renderConfig.width);
			encoder.setHeight(renderConfig.height);

			PixelFormat.Type pixelformat = PixelFormat.Type.PIX_FMT_YUV420P;
			encoder.setPixelFormat(pixelformat);
			encoder.setTimeBase(Rational.make(1.0 / 25.0));

			if (format.getFlag(MuxerFormat.Flag.GLOBAL_HEADER)) {
				encoder.setFlag(Encoder.Flag.FLAG_GLOBAL_HEADER, true);
			}

			encoder.open(null, null);
			muxer.addNewStream(encoder);

			muxer.open(null, null);

			MediaPicture picture = MediaPicture.make(encoder.getWidth(), encoder.getHeight(), pixelformat);
			picture.setTimeBase(Rational.make(1.0 / 25.0));

			MediaPacket packet = MediaPacket.make();
			MediaPictureConverter converter = null;

			double time = renderConfig.startTime;
			int frameIndex = 0;

			while (time < renderConfig.endTime) {
				System.out.println(Time.writeTime(time) + " / " + Time.writeTime(renderConfig.endTime));

				BufferedImage image = new BufferedImage(renderConfig.width, renderConfig.height,
						BufferedImage.TYPE_3BYTE_BGR);

				renderer.update((Graphics2D) image.getGraphics(), time);

				do {
					BufferedImage newImage = new BufferedImage(renderConfig.width, renderConfig.height,
							BufferedImage.TYPE_3BYTE_BGR);
					newImage.getGraphics().drawImage(image, 0, 0, null);

					if (converter == null) {
						converter = MediaPictureConverterFactory.createConverter(newImage, picture);
					}

					converter.toPicture(picture, newImage, frameIndex);

					encoder.encode(packet, picture);
					if (packet.isComplete()) {
						muxer.write(packet, false);
					}
				} while (packet.isComplete());

				time += renderConfig.secondsPerFrame / 25.0;
				frameIndex++;
			}

			do {
				encoder.encode(packet, null);

				if (packet.isComplete()) {
					muxer.write(packet, false);
				}
			} while (packet.isComplete());

			muxer.close();
		} catch (InterruptedException | IOException e) {
			throw new RuntimeException(e);
		}
	}
}
