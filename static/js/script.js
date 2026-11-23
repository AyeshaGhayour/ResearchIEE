function openFeature(feature) {

    const featureNames = {
        explorer: "Semantic Research Explorer",
        connections: "Research Connection Map",
        comparison: "Paper Comparison Engine",
        gaps: "Research Gap Explorer",
        ideas: "Idea Catalyst"
    };

    const featureName = featureNames[feature];

    if (!featureName) {
        console.error("Unknown feature:", feature);
        return;
    }

    alert(featureName + " workspace is coming soon!");

}
