package ch.ethzm.matsim.renderer;

import java.io.File;
import java.io.IOException;

import com.fasterxml.jackson.core.JsonGenerationException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import ch.ethzm.matsim.renderer.config.RenderConfig;
import ch.ethzm.matsim.renderer.main.RunRenderer;

public class RunVisualization {
	static public void main(String[] args) throws JsonGenerationException, JsonMappingException, IOException {
		/*-
		 * This script writes a series of png files to an output directory,
		 * visualizing a MATSim simulation. As input it needs a network file and an
		 * events file from the MATSim output. Afterwards, using FFMPEG the png files
		 * can be combined to a video with a command like this:
		 * 
		 * ffmpeg -framerate 25 -i video_%d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4
		 */

		RenderConfig renderConfig = new ObjectMapper().readValue(new File(args[0]), RenderConfig.class);
		RunRenderer.run(renderConfig);
	}
}
