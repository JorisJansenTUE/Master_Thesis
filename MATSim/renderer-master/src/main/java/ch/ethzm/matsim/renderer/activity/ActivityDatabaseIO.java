package ch.ethzm.matsim.renderer.activity;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.List;

public class ActivityDatabaseIO {
	static public void write(File path, ActivityDatabase database) throws IOException {
		DataOutputStream writer = new DataOutputStream(new FileOutputStream(path));

		for (List<ActivitySlice> slices : database.getData()) {
			for (ActivitySlice slice : slices) {
				writer.writeDouble(slice.startTime);
				writer.writeDouble(slice.endTime);
				writer.writeInt(slice.typeIndex);
				writer.writeInt(slice.linkIndex);
			}
		}

		writer.close();
	}

	static public void read(File path, ActivityDatabase database) throws IOException {
		DataInputStream reader = new DataInputStream(new FileInputStream(path));

		while (reader.available() > 0) {
			double startTime = reader.readDouble();
			double endTime = reader.readDouble();
			int typeIndex = reader.readInt();
			int linkIndex = reader.readInt();

			database.addActivity(new ActivitySlice(startTime, endTime, typeIndex, linkIndex));
		}

		reader.close();
	}
}
