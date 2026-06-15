package ch.ethzm.matsim.renderer.activity;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

public class ActivityTypeIO {
	static public void write(File path, ActivityTypeMapper mapper) throws IOException {
		DataOutputStream writer = new DataOutputStream(new FileOutputStream(path));

		for (String activityType : mapper.getData()) {
			writer.writeUTF(activityType);
		}

		writer.close();
	}

	static public void read(File path, ActivityTypeMapper mapper) throws IOException {
		DataInputStream reader = new DataInputStream(new FileInputStream(path));

		while (reader.available() > 0) {
			String activityType = reader.readUTF();
			mapper.addActivityType(activityType);
		}

		reader.close();
	}
}
