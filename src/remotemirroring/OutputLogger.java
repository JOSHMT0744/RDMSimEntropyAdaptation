package remotemirroring;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Generic helper for appending whitespace-separated rows of monitored output
 * to flat text files in the configured output directory, following the same
 * convention used by DeltaIOTConnector in the L4Project DeltaIoT port:
 * headerless "timestep value..." rows appended per write, with an optional
 * one-time header for multi-column files.
 */
public class OutputLogger {
	private static final Logger log = LogManager.getLogger(OutputLogger.class);

	private final String outputDirectory;

	public OutputLogger(String outputDirectory) {
		this.outputDirectory = outputDirectory;
	}

	/** Truncates (or creates) the given file, ready for a fresh run. */
	public void clearFile(String filename) {
		File file = new File(outputDirectory, filename);
		try (FileWriter fw = new FileWriter(file, false)) {
			// Opening with append=false truncates the file
		} catch (IOException e) {
			log.error("Error clearing {}: {}", filename, e.getMessage());
		}
	}

	/** Appends a single whitespace-separated row of values to filename. */
	public void appendRow(String filename, Object... values) {
		File file = new File(outputDirectory, filename);
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < values.length; i++) {
			if (i > 0) sb.append(' ');
			sb.append(values[i]);
		}
		try (BufferedWriter writer = new BufferedWriter(new FileWriter(file, true))) {
			writer.write(sb.toString());
			writer.newLine();
		} catch (IOException e) {
			log.error("Error writing to {}: {}", filename, e.getMessage());
		}
	}

	/** Writes the header row for monitorables.txt if the file doesn't already contain it (call once after clearFile). */
	public void writeMonitorablesHeader() {
		appendRow("monitorables.txt", "timestep", "active_links", "bandwidth_consumption", "time_to_write",
				"alpha", "threshold_active_links", "threshold_bandwidth_consumption", "threshold_time_to_write");
	}
}
