package thesis.population;

import org.geotools.api.feature.simple.SimpleFeature;
import org.locationtech.jts.geom.*;
import org.matsim.api.core.v01.Coord;
import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.TransportMode;
import org.matsim.api.core.v01.network.Network;
import org.matsim.api.core.v01.population.*;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.network.io.MatsimNetworkReader;
import org.matsim.core.population.PopulationUtils;
import org.matsim.core.population.io.PopulationWriter;
import org.matsim.core.scenario.ScenarioUtils;
import org.matsim.core.utils.gis.ShapeFileReader;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;

/**
 * Minimal synthetic population generator for Locarno.
 *
 * Reads one polygon shapefile for the study area, samples random home/work
 * locations inside it, creates home-work-home plans, assigns bike/car by
 * distance, and writes a MATSim population file.
 */
public class CreateSyntheticPopulationFromShapefile {

    // ----------------------------
    // INPUT / OUTPUT
    // ----------------------------
    private static final String NETWORK_FILE = "../../data/network/locarno_multimodal_v1.xml.gz";
    private static final String SHAPE_FILE = "../../data/osm/shapes/Locarno_QGIS_2056.shp";
    private static final String OUTPUT_POPULATION_FILE = "../../data/population/synthetic_population1000_multi.xml";

    // ----------------------------
    // SYNTHETIC POPULATION SETTINGS
    // ----------------------------
    private static final int N_PERSONS = 1000;
    private static final double BIKE_DISTANCE_THRESHOLD_M = 2000.0; //was 4000 for 10person run

    // Times in seconds
    private static final double HOME_END_TIME_MIN = 7.0 * 3600.0;   // 07:00
    private static final double HOME_END_TIME_MAX = 9.0 * 3600.0;   // 09:00
    private static final double WORK_END_TIME_MIN = 16.0 * 3600.0;  // 16:00
    private static final double WORK_END_TIME_MAX = 18.0 * 3600.0;  // 18:00

    private final Random random = new Random(1234);
    private final GeometryFactory geometryFactory = new GeometryFactory();

    private final List<Geometry> studyAreaGeometries = new ArrayList<>();

    private Population population;

    public static void main(String[] args) {
        new CreateSyntheticPopulationFromShapefile().run();
    }

    private void run() {
        var config = ConfigUtils.createConfig();
        var scenario = ScenarioUtils.createScenario(config);

        // Load network
        new MatsimNetworkReader(scenario.getNetwork()).readFile(NETWORK_FILE);
        Network network = scenario.getNetwork();

        // Create population container
        population = PopulationUtils.createPopulation(ConfigUtils.createConfig());

        // Load polygon shapefile
        loadStudyArea(SHAPE_FILE);

        if (studyAreaGeometries.isEmpty()) {
            throw new RuntimeException("No geometries found in shapefile: " + SHAPE_FILE);
        }

        // Generate synthetic persons
        for (int i = 0; i < N_PERSONS; i++) {
            Coord home = sampleCoordInStudyArea();
            Coord work = sampleCoordInStudyArea();

            double distance = euclideanDistance(home, work);
            String mode = distance < BIKE_DISTANCE_THRESHOLD_M ? TransportMode.bike : TransportMode.car;

            Person person = createPerson(home, work, mode, "person_" + i);
            population.addPerson(person);
        }

        // Write output
        new PopulationWriter(population, network).write(OUTPUT_POPULATION_FILE);

        System.out.println("Done.");
        System.out.println("Population written to: " + OUTPUT_POPULATION_FILE);
        System.out.println("Persons created: " + population.getPersons().size());
    }

    private void loadStudyArea(String shapefile) {
        Collection<SimpleFeature> features = ShapeFileReader.getAllFeatures(shapefile);

        for (SimpleFeature feature : features) {
            Object geomObj = feature.getDefaultGeometry();
            if (geomObj instanceof Geometry geometry) {
                studyAreaGeometries.add(geometry);
            }
        }
    }

    private Person createPerson(Coord home, Coord work, String mode, String id) {
        PopulationFactory factory = population.getFactory();

        Person person = factory.createPerson(Id.createPersonId(id));
        Plan plan = createPlan(home, work, mode);

        person.addPlan(plan);
        person.setSelectedPlan(plan);

        return person;
    }

    private Plan createPlan(Coord home, Coord work, String mode) {
        PopulationFactory factory = population.getFactory();
        Plan plan = factory.createPlan();

        double homeEndTime = sampleUniform(HOME_END_TIME_MIN, HOME_END_TIME_MAX);
        double workEndTime = sampleUniform(WORK_END_TIME_MIN, WORK_END_TIME_MAX);

        // home
        Activity homeMorning = factory.createActivityFromCoord("home", home);
        homeMorning.setEndTime(homeEndTime);
        plan.addActivity(homeMorning);

        // leg to work
        Leg toWork = factory.createLeg(mode);
        plan.addLeg(toWork);

        // work
        Activity workActivity = factory.createActivityFromCoord("work", work);
        workActivity.setEndTime(workEndTime);
        plan.addActivity(workActivity);

        // leg to home
        Leg toHome = factory.createLeg(mode);
        plan.addLeg(toHome);

        // final home
        Activity homeEvening = factory.createActivityFromCoord("home", home);
        plan.addActivity(homeEvening);

        return plan;
    }

    private Coord sampleCoordInStudyArea() {
        // Pick one geometry at random
        Geometry geometry = studyAreaGeometries.get(random.nextInt(studyAreaGeometries.size()));
        return sampleCoordInGeometry(geometry);
    }

    private Coord sampleCoordInGeometry(Geometry geometry) {
        Envelope envelope = geometry.getEnvelopeInternal();

        double x;
        double y;
        Point point;

        do {
            x = envelope.getMinX() + envelope.getWidth() * random.nextDouble();
            y = envelope.getMinY() + envelope.getHeight() * random.nextDouble();
            point = geometryFactory.createPoint(new Coordinate(x, y));
        } while (!geometry.contains(point));

        return new Coord(x, y);
    }

    private double euclideanDistance(Coord a, Coord b) {
        double dx = a.getX() - b.getX();
        double dy = a.getY() - b.getY();
        return Math.sqrt(dx * dx + dy * dy);
    }

    private double sampleUniform(double min, double max) {
        return min + random.nextDouble() * (max - min);
    }
}