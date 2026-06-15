package ch.ethzm.matsim.renderer.config;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import com.fasterxml.jackson.annotation.JsonIgnore;

public class VehicleConfig {
	public List<String> startsWith = new LinkedList<>();
	public List<String> endsWith = new LinkedList<>();
	public List<String> contains = new LinkedList<>();

	public List<Integer> color = Arrays.asList(200, 200, 200);
	public int size = 4;

	public void validate() {
		if (color.size() != 3) {
			throw new IllegalStateException("Color must be (R,G,B)");
		}
	}

	@JsonIgnore
	public boolean isGeneric() {
		return startsWith.size() == 0 && endsWith.size() == 0 && contains.size() == 0;
	}
}
