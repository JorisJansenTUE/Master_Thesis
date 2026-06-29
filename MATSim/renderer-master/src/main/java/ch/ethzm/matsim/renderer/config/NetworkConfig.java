package ch.ethzm.matsim.renderer.config;

import java.util.Arrays;
import java.util.List;

public class NetworkConfig {
	public List<String> modes = Arrays.asList("car");
	public List<Integer> color = Arrays.asList(200, 200, 200);

	public void validate() {
		if (color.size() != 3) {
			throw new IllegalStateException("Color must be (R,G,B)");
		}
	}
}
