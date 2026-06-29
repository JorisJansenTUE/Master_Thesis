package ch.ethzm.matsim.renderer.traversal;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.List;

public class TraversalDatabaseIO {
	static public void write(File path, TraversalDatabase database) throws IOException {
		DataOutputStream writer = new DataOutputStream(new FileOutputStream(path));

		for (List<Traversal> traversals : database.getData()) {
			for (Traversal traversal : traversals) {
				writer.writeDouble(traversal.startTime);
				writer.writeDouble(traversal.endTime);
				writer.writeInt(traversal.linkIndex);
				writer.writeInt(traversal.vehicleIndex);
				writer.writeInt(traversal.vehicleType);
			}
		}

		writer.close();
	}

	static public void read(File path, TraversalDatabase database) throws IOException {
		DataInputStream reader = new DataInputStream(new FileInputStream(path));

		while (reader.available() > 0) {
			double startTime = reader.readDouble();
			double endTime = reader.readDouble();
			int linkIndex = reader.readInt();
			int vehicleIndex = reader.readInt();
			int typeIndex = reader.readInt();

			database.addTraversal(new Traversal(linkIndex, -1, vehicleIndex, startTime, endTime, typeIndex));
		}

		reader.close();
	}
}
