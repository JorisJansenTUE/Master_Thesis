package thesis.network;

import java.util.Set;

import org.matsim.api.core.v01.Scenario;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.network.NetworkUtils;
import org.matsim.core.network.io.MatsimNetworkReader;
import org.matsim.core.network.io.NetworkWriter;
import org.matsim.core.scenario.ScenarioUtils;

public class CleanModeSubnetwork {
    public static void main(String[] args) {
        if (args.length != 3) {
            throw new IllegalArgumentException(
                "Usage: CleanModeSubnetwork <inputNetwork> <outputNetwork> <mode>"
            );
        }

        String inputNetwork = args[0];
        String outputNetwork = args[1];
        String mode = args[2];

        Scenario scenario = ScenarioUtils.createScenario(ConfigUtils.createConfig());

        new MatsimNetworkReader(scenario.getNetwork()).readFile(inputNetwork);

        long before = scenario.getNetwork().getLinks().values().stream()
            .filter(link -> link.getAllowedModes().contains(mode))
            .count();

        NetworkUtils.cleanNetwork(
            scenario.getNetwork(),
            Set.of(mode)
        );

        long after = scenario.getNetwork().getLinks().values().stream()
            .filter(link -> link.getAllowedModes().contains(mode))
            .count();

        System.out.println("Mode: " + mode);
        System.out.println("Links before cleaning: " + before);
        System.out.println("Links after cleaning:  " + after);
        System.out.println("Mode removed from:     " + (before - after));

        new NetworkWriter(scenario.getNetwork()).write(outputNetwork);
    }
}