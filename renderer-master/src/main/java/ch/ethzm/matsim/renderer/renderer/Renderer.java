package ch.ethzm.matsim.renderer.renderer;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;

import java.io.File;
import javax.imageio.ImageIO;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.Map;

import org.matsim.api.core.v01.Coord;
import org.matsim.api.core.v01.network.Link;
import org.matsim.core.utils.geometry.CoordUtils;
import org.matsim.core.utils.misc.Time;

import ch.ethzm.matsim.renderer.activity.ActivityDatabase;
import ch.ethzm.matsim.renderer.activity.ActivityTypeMapper;
import ch.ethzm.matsim.renderer.config.ActivityConfig;
import ch.ethzm.matsim.renderer.config.NetworkConfig;
import ch.ethzm.matsim.renderer.config.RenderConfig;
import ch.ethzm.matsim.renderer.config.VehicleConfig;
import ch.ethzm.matsim.renderer.main.Transform;
import ch.ethzm.matsim.renderer.network.LinkDatabase;
import ch.ethzm.matsim.renderer.traversal.TraversalDatabase;
import ch.ethzm.matsim.renderer.traversal.VehicleDatabase;

public class Renderer {
	private final int windowWidth;
	private final int windowHeight;

	private final RenderConfig renderConfig;
	private final BufferedImage backgroundImage;
	
	final private TraversalDatabase traversalDatabase;
	final private LinkDatabase linkDatabase;
	final private ActivityDatabase activityDatabase;

	final private ActivityTypeMapper activityTypeMapper;
	final private VehicleDatabase vehicleDatabase;

	private final List<Color> linkColors;
	private final List<Color> vehicleColors;
	private final List<Integer> vehicleSizes;
	private final List<Color> activityColors;
	private final List<Double> activityMaximumLifetimes;
	private final List<Double> activitySizes;

	final private BitSet isAv;
	final private BitSet isPt;
	final private BitSet isFreight;

	public Renderer(int windowWidth, int windowHeight, RenderConfig renderConfig, TraversalDatabase traversalDatabase,
			LinkDatabase linkDatabase, ActivityDatabase activityDatabase, ActivityTypeMapper activityTypeMapper,
			VehicleDatabase vehicleDatabase) {
		this.windowWidth = windowWidth;
		this.windowHeight = windowHeight;

		this.renderConfig = renderConfig;
		this.traversalDatabase = traversalDatabase;
		this.linkDatabase = linkDatabase;
		this.activityDatabase = activityDatabase;
		this.activityTypeMapper = activityTypeMapper;
		this.vehicleDatabase = vehicleDatabase;

		try {
			if (renderConfig.backgroundImagePath != null) {
				this.backgroundImage = ImageIO.read(new File(renderConfig.backgroundImagePath));
			} else {
				this.backgroundImage = null;
			}
		} catch (Exception e) {
			throw new RuntimeException("Could not load background image: " + renderConfig.backgroundImagePath, e);
		}

		windowWidth = renderConfig.width;
		windowHeight = renderConfig.height;

		this.isAv = new BitSet(vehicleDatabase.getVehicleIndices().size());
		this.isPt = new BitSet(vehicleDatabase.getVehicleIndices().size());
		this.isFreight = new BitSet(vehicleDatabase.getVehicleIndices().size());

		for (Map.Entry<String, Integer> entry : vehicleDatabase.getVehicleIndices().entrySet()) {
			if (entry.getKey().contains("av")) {
				isAv.set(entry.getValue());
			}

			if (entry.getKey().contains("subway")) {
				isPt.set(entry.getValue());
			}

			if (entry.getKey().contains("freight")) {
				isFreight.set(entry.getValue());
			}
		}

		this.linkColors = new ArrayList<>(renderConfig.networks.size());

		for (NetworkConfig networkConfig : renderConfig.networks) {
			linkColors
					.add(new Color(networkConfig.color.get(0), networkConfig.color.get(1), networkConfig.color.get(2)));
		}

		this.vehicleColors = new ArrayList<>(renderConfig.vehicles.size());

		for (VehicleConfig vehicleConfig : renderConfig.vehicles) {
			vehicleColors
					.add(new Color(vehicleConfig.color.get(0), vehicleConfig.color.get(1), vehicleConfig.color.get(2)));
		}

		this.vehicleSizes = new ArrayList<>(renderConfig.vehicles.size());

		for (VehicleConfig vehicleConfig : renderConfig.vehicles) {
			vehicleSizes.add(vehicleConfig.size);
		}

		this.activityColors = new ArrayList<>(renderConfig.activities.size());
		this.activityMaximumLifetimes = new ArrayList<>(renderConfig.activities.size());
		this.activitySizes = new ArrayList<>(renderConfig.activities.size());

		for (ActivityConfig activtiyConfig : renderConfig.activities) {
			activityColors.add(
					new Color(activtiyConfig.color.get(0), activtiyConfig.color.get(1), activtiyConfig.color.get(2)));
			activityMaximumLifetimes.add(activtiyConfig.maximumLifetime);
			activitySizes.add(activtiyConfig.size);
		}
	}

	public void update(Graphics2D graphics, double time) {
		if (backgroundImage != null) {
			graphics.drawImage(backgroundImage, 0, 0, windowWidth, windowHeight, null);
		} else {
			graphics.setColor(Color.WHITE);
			graphics.fillRect(0, 0, windowWidth, windowHeight);
		}
		graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

		Coord center = new Coord(renderConfig.center.get(0), renderConfig.center.get(1));
		double zoom = renderConfig.zoom;

		double f = (double) windowHeight / (double) windowWidth;

		double minx = center.getX() - zoom;
		double maxx = center.getX() + zoom;
		double miny = center.getY() - zoom * f;
		double maxy = center.getY() + zoom * f;

		Rectangle scenarioBounds = new Rectangle((int) minx, (int) miny, (int) (maxx - minx), (int) (maxy - miny));
		Rectangle windowBounds = new Rectangle(0, 0, windowWidth, windowHeight);
		Transform transform = new Transform(windowBounds, scenarioBounds);

		linkDatabase.getLinks().stream()
				.filter(l -> scenarioBounds.contains(l.link.getCoord().getX(), l.link.getCoord().getY()))
				.filter(l -> l.typeIndex >= 0).forEach(l -> {
					graphics.setColor(linkColors.get(l.typeIndex));

					Coord fromCoord = transform.scenarioToWindow(l.link.getFromNode().getCoord());
					Coord toCoord = transform.scenarioToWindow(l.link.getToNode().getCoord());

					graphics.drawLine((int) fromCoord.getX(), (int) fromCoord.getY(), (int) toCoord.getX(),
							(int) toCoord.getY());
				});

		traversalDatabase.getTraversalsAtTime(time)
				.filter(t -> scenarioBounds.contains(linkDatabase.getLink(t.linkIndex).link.getCoord().getX(),
						linkDatabase.getLink(t.linkIndex).link.getCoord().getY()))
				.filter(t -> t.vehicleType >= 0).sequential().forEach(t -> {
					graphics.setColor(vehicleColors.get(t.vehicleType));

					Link link = linkDatabase.getLink(t.linkIndex).link;

					double s = (time - t.startTime) / (t.endTime - t.startTime);

					Coord coord = CoordUtils.plus(link.getFromNode().getCoord(), CoordUtils.scalarMult(s,
							CoordUtils.minus(link.getToNode().getCoord(), link.getFromNode().getCoord())));

					coord = transform.scenarioToWindow(coord);

					int size = vehicleSizes.get(t.vehicleType);
					graphics.fillOval((int) coord.getX() - size / 2, (int) coord.getY() - size / 2, size, size);
				});

		activityDatabase.getActivitiesAtTime(time)
				.filter(t -> scenarioBounds.contains(linkDatabase.getLink(t.linkIndex).link.getCoord().getX(),
						linkDatabase.getLink(t.linkIndex).link.getCoord().getY()))
				.filter(t -> t.typeIndex >= 0) //
				.sequential().forEach(a -> {
					Link link = linkDatabase.getLink(a.linkIndex).link;
					Coord coord = transform.scenarioToWindow(link.getCoord());

					double lifetime = time - a.startTime;

					double activityLifetime = activityMaximumLifetimes.get(a.typeIndex);
					double maximumLifetime = Math.min(activityLifetime, a.endTime - a.startTime);

					if (lifetime <= activityLifetime) {
						double intensity = lifetime / maximumLifetime;

						double size = activitySizes.get(a.typeIndex) * intensity;
						double alpha = Math.floor(200.0 - 200.0 * intensity);
						Color color = activityColors.get(a.typeIndex);

						graphics.setColor(new Color(color.getRed(), color.getGreen(), color.getBlue(), (int) alpha));

						graphics.fillOval((int) (coord.getX() - 0.5 * size), (int) (coord.getY() - 0.5 * size),
								(int) size, (int) size);
					}
				});

		if (renderConfig.showTime) {
			graphics.setColor(Color.BLACK);
			graphics.setFont(new Font(graphics.getFont().getName(), Font.BOLD, 20));
			graphics.drawString(Time.writeTime(time), 40, 40);
		}
	}
}
