package ch.ethzm.matsim.renderer.traversal;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

public class VehicleDatabaseIO {
	static public void write(File path, VehicleDatabase vehicleDatabase) throws IOException {
		DataOutputStream writer = new DataOutputStream(new FileOutputStream(path));

		for (String id : vehicleDatabase.getData()) {
			writer.writeUTF(id);
		}

		writer.close();
	}

	static public void read(File path, VehicleDatabase vehicleDatabase) throws IOException {
		DataInputStream reader = new DataInputStream(new FileInputStream(path));

		while (reader.available() > 0) {
			String id = reader.readUTF();
			vehicleDatabase.addVehicle(id);
		}

		reader.close();
	}
}
