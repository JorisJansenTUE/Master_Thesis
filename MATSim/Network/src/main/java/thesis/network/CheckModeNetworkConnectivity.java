package thesis.network;

import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.network.Link;
import org.matsim.api.core.v01.network.Network;
import org.matsim.api.core.v01.network.Node;
import org.matsim.core.network.NetworkUtils;
import org.matsim.core.network.algorithms.NetworkCleaner;
import org.matsim.core.network.algorithms.TransportModeNetworkFilter;
import org.matsim.core.network.io.NetworkWriter;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.Set;

public class CheckModeNetworkConnectivity {
    public static void main(String[] args) throws IOException {
        if (args.length != 4) {
            throw new IllegalArgumentException(
                    "Usage: CheckModeNetworkConnectivity <inputNetwork> <mode> <outputRemovedCsv> <outputCleanedNetwork>"
            );
        }

        String inputNetworkPath = args[0];
        String mode = args[1];
        Path outputRemovedCsv = Path.of(args[2]);
        String outputCleanedNetworkPath = args[3];

        Network fullNetwork = NetworkUtils.readNetwork(inputNetworkPath);

        Network modeNetwork = NetworkUtils.createNetwork();
        new TransportModeNetworkFilter(fullNetwork).filter(modeNetwork, Set.of(mode));

        Set<Id<Link>> beforeLinks = new HashSet<>(modeNetwork.getLinks().keySet());
        Set<Id<Node>> beforeNodes = new HashSet<>(modeNetwork.getNodes().keySet());

        System.out.println("Full network links: " + fullNetwork.getLinks().size());
        System.out.println("Mode network links for '" + mode + "': " + modeNetwork.getLinks().size());
        System.out.println("Mode network nodes for '" + mode + "': " + modeNetwork.getNodes().size());

        new NetworkCleaner().run(modeNetwork);

        Set<Id<Link>> afterLinks = new HashSet<>(modeNetwork.getLinks().keySet());
        Set<Id<Node>> afterNodes = new HashSet<>(modeNetwork.getNodes().keySet());

        beforeLinks.removeAll(afterLinks);
        beforeNodes.removeAll(afterNodes);

        System.out.println("Removed unreachable links: " + beforeLinks.size());
        System.out.println("Removed unreachable nodes: " + beforeNodes.size());

        Files.createDirectories(outputRemovedCsv.getParent());

        try (BufferedWriter writer = Files.newBufferedWriter(outputRemovedCsv)) {
            writer.write("link_id,from_node,to_node,from_x,from_y,to_x,to_y,length,freespeed,allowed_modes,origid,type\n");

            for (Id<Link> linkId : beforeLinks) {
                Link link = fullNetwork.getLinks().get(linkId);

                if (link == null) {
                    continue;
                }

                Node from = link.getFromNode();
                Node to = link.getToNode();

                Object origId = link.getAttributes().getAttribute("origid");
                Object type = link.getAttributes().getAttribute("type");

                writer.write(String.format(
                        "%s,%s,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,\"%s\",%s,%s%n",
                        link.getId(),
                        from.getId(),
                        to.getId(),
                        from.getCoord().getX(),
                        from.getCoord().getY(),
                        to.getCoord().getX(),
                        to.getCoord().getY(),
                        link.getLength(),
                        link.getFreespeed(),
                        link.getAllowedModes(),
                        origId == null ? "" : origId,
                        type == null ? "" : type
                ));
            }
        }

        new NetworkWriter(modeNetwork).write(outputCleanedNetworkPath);

        if (!beforeLinks.isEmpty() || !beforeNodes.isEmpty()) {
            throw new RuntimeException(
                    "Network for mode '" + mode + "' is not strongly connected. " +
                    "See removed links CSV: " + outputRemovedCsv
            );
        }

        System.out.println("Network for mode '" + mode + "' is strongly connected.");
    }
}