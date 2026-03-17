import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../theme"
import "../components"

Item {
    id: defectPage

    property bool hasResult: false
    property bool inferring: false
    property int queryVersion: 0
    property int attentionVersion: 0
    property int overlayVersion: 0

    Connections {
        target: bridge

        function onDefectAnalysisReady() {
            hasResult = true
            inferring = false
        }

        function onInferenceStarted(model) {
            if (model === "defect") inferring = true
        }

        function onInferenceFinished(model) {
            if (model === "defect") inferring = false
        }

        function onImageUpdated(imageId, version) {
            if (imageId === "defect_query") queryVersion = version
            else if (imageId === "defect_attention") attentionVersion = version
            else if (imageId === "defect_overlay") overlayVersion = version
        }
    }

    Flickable {
        anchors.fill: parent
        anchors.margins: 24
        contentWidth: width
        contentHeight: mainCol.implicitHeight
        clip: true
        boundsBehavior: Flickable.StopAtBounds

        ColumnLayout {
            id: mainCol
            width: parent.width
            spacing: 20

            // --- HEADER ---
            RowLayout {
                Layout.fillWidth: true
                spacing: 16

                ColumnLayout {
                    spacing: 4
                    Text {
                        text: "Defect Detection & Analysis"
                        font.pixelSize: Theme.fontHero
                        font.bold: true
                        color: Theme.textPrimary
                    }
                    Text {
                        text: "Drop a wafer image to detect defects, view attention maps, and get analysis"
                        font.pixelSize: Theme.fontBody
                        color: Theme.textSecondary
                    }
                }

                Item { Layout.fillWidth: true }

                // Support set status badge
                Rectangle {
                    width: supportRow.width + 24
                    height: 36
                    radius: 18
                    color: bridge.supportLoaded ? Theme.success + "20" :
                           bridge.supportLoading ? Theme.warning + "20" : Theme.textDim + "20"

                    RowLayout {
                        id: supportRow
                        anchors.centerIn: parent
                        spacing: 8

                        Rectangle {
                            width: 8; height: 8; radius: 4
                            color: bridge.supportLoaded ? Theme.success :
                                   bridge.supportLoading ? Theme.warning : Theme.textDim

                            SequentialAnimation on opacity {
                                running: bridge.supportLoading
                                loops: Animation.Infinite
                                NumberAnimation { to: 0.3; duration: 500 }
                                NumberAnimation { to: 1.0; duration: 500 }
                            }
                        }

                        Text {
                            text: bridge.supportLoading ? "Loading prototypes..." :
                                  bridge.supportLoaded ? bridge.supportClassCount + " classes ready (" + bridge.supportImageCount + " images)" :
                                  "Support not loaded"
                            font.pixelSize: Theme.fontSmall
                            color: bridge.supportLoaded ? Theme.success :
                                   bridge.supportLoading ? Theme.warning : Theme.textDim
                        }
                    }
                }
            }

            // --- DROP ZONE ---
            ImageDropZone {
                Layout.fillWidth: true
                Layout.preferredHeight: 120
                title: "Drop Wafer Image to Analyze"
                subtitle: "PNG, JPG, BMP, TIFF supported"

                onFileSelected: function(path) {
                    bridge.analyzeDefect(path)
                }
            }

            // --- ANALYZING INDICATOR ---
            Rectangle {
                Layout.fillWidth: true
                height: 52
                radius: Theme.cardRadius
                color: Theme.accent + "15"
                border.color: Theme.accent + "40"
                border.width: 1
                visible: inferring

                RowLayout {
                    anchors.centerIn: parent
                    spacing: 12

                    BusyIndicator {
                        running: true
                        implicitWidth: 24
                        implicitHeight: 24
                    }

                    Text {
                        text: "Analyzing image with DINOv2 attention..."
                        font.pixelSize: Theme.fontBody
                        color: Theme.accent
                    }
                }
            }

            // --- SEVERITY BANNER ---
            Rectangle {
                visible: hasResult
                Layout.fillWidth: true
                height: 64
                radius: Theme.cardRadius
                color: {
                    var s = bridge.defectSeverity
                    if (s === "critical") return Theme.error + "25"
                    if (s === "high") return Theme.error + "18"
                    if (s === "medium") return Theme.warning + "18"
                    return Theme.success + "18"
                }
                border.color: {
                    var s = bridge.defectSeverity
                    if (s === "critical") return Theme.error
                    if (s === "high") return Theme.error + "80"
                    if (s === "medium") return Theme.warning + "80"
                    return Theme.success + "80"
                }
                border.width: 1

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 20
                    anchors.rightMargin: 20
                    spacing: 16

                    // Severity badge
                    Rectangle {
                        width: sevText.width + 20
                        height: 28
                        radius: 14
                        color: {
                            var s = bridge.defectSeverity
                            if (s === "critical") return Theme.error
                            if (s === "high") return Theme.error + "CC"
                            if (s === "medium") return Theme.warning
                            return Theme.success
                        }

                        Text {
                            id: sevText
                            anchors.centerIn: parent
                            text: bridge.defectSeverity ? bridge.defectSeverity.toUpperCase() : ""
                            font.pixelSize: Theme.fontSmall
                            font.bold: true
                            color: bridge.defectSeverity === "medium" ? Theme.background : "#ffffff"
                        }
                    }

                    Text {
                        text: bridge.defectName
                        font.pixelSize: Theme.fontHeader
                        font.bold: true
                        color: Theme.textPrimary
                        Layout.fillWidth: true
                        elide: Text.ElideRight
                    }

                    Text {
                        text: "Class " + bridge.predictedDefectClass + "  |  " +
                              (bridge.defectConfidence * 100).toFixed(1) + "% confidence"
                        font.pixelSize: Theme.fontBody
                        color: Theme.textSecondary
                    }
                }
            }

            // --- IMAGE ROW ---
            RowLayout {
                visible: hasResult
                Layout.fillWidth: true
                Layout.preferredHeight: 320
                spacing: 16

                HeatmapViewer {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    imageId: "defect_query"
                    imageVersion: queryVersion
                    label: "Original Image"
                }

                HeatmapViewer {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    imageId: "defect_attention"
                    imageVersion: attentionVersion
                    label: "Attention Map"
                }

                HeatmapViewer {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    imageId: "defect_overlay"
                    imageVersion: overlayVersion
                    label: "Defect Localization"
                }
            }

            // --- DESCRIPTION + CAUSES / PREVENTION ---
            RowLayout {
                visible: hasResult
                Layout.fillWidth: true
                spacing: 16

                // Left card: Description + Root Causes
                Rectangle {
                    Layout.fillWidth: true
                    implicitHeight: descCol.implicitHeight + 32
                    radius: Theme.cardRadius
                    color: Theme.surface
                    border.color: Theme.divider
                    border.width: 1

                    ColumnLayout {
                        id: descCol
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.top: parent.top
                        anchors.margins: 16
                        spacing: 12

                        Text {
                            text: "Defect Description"
                            font.pixelSize: Theme.fontTitle
                            font.bold: true
                            color: Theme.textPrimary
                        }

                        Text {
                            text: bridge.defectDescription
                            font.pixelSize: Theme.fontBody
                            color: Theme.textSecondary
                            wrapMode: Text.Wrap
                            Layout.fillWidth: true
                            lineHeight: 1.5
                        }

                        Rectangle {
                            Layout.fillWidth: true
                            height: 1
                            color: Theme.divider
                            Layout.topMargin: 4
                            Layout.bottomMargin: 4
                        }

                        Text {
                            text: "Root Causes"
                            font.pixelSize: Theme.fontBody
                            font.bold: true
                            color: Theme.error
                        }

                        Repeater {
                            model: bridge.defectCauses

                            RowLayout {
                                spacing: 10
                                Layout.fillWidth: true

                                Text {
                                    text: "\u2022"
                                    color: Theme.error
                                    font.pixelSize: Theme.fontBody
                                    Layout.alignment: Qt.AlignTop
                                }

                                Text {
                                    text: modelData
                                    font.pixelSize: Theme.fontSmall
                                    color: Theme.textSecondary
                                    wrapMode: Text.Wrap
                                    Layout.fillWidth: true
                                    lineHeight: 1.4
                                }
                            }
                        }
                    }
                }

                // Right card: Prevention & Mitigation
                Rectangle {
                    Layout.fillWidth: true
                    implicitHeight: prevCol.implicitHeight + 32
                    radius: Theme.cardRadius
                    color: Theme.surface
                    border.color: Theme.divider
                    border.width: 1

                    ColumnLayout {
                        id: prevCol
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.top: parent.top
                        anchors.margins: 16
                        spacing: 12

                        Text {
                            text: "Prevention & Mitigation"
                            font.pixelSize: Theme.fontTitle
                            font.bold: true
                            color: Theme.success
                        }

                        Repeater {
                            model: bridge.defectPrevention

                            RowLayout {
                                spacing: 10
                                Layout.fillWidth: true

                                Text {
                                    text: "\u2713"
                                    color: Theme.success
                                    font.pixelSize: Theme.fontBody
                                    font.bold: true
                                    Layout.alignment: Qt.AlignTop
                                }

                                Text {
                                    text: modelData
                                    font.pixelSize: Theme.fontSmall
                                    color: Theme.textSecondary
                                    wrapMode: Text.Wrap
                                    Layout.fillWidth: true
                                    lineHeight: 1.4
                                }
                            }
                        }
                    }
                }
            }

            // --- CLASSIFICATION DETAILS ---
            Rectangle {
                visible: hasResult
                Layout.fillWidth: true
                implicitHeight: probCol.implicitHeight + 32
                radius: Theme.cardRadius
                color: Theme.surface
                border.color: Theme.divider
                border.width: 1

                ColumnLayout {
                    id: probCol
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.top: parent.top
                    anchors.margins: 16
                    spacing: 8

                    RowLayout {
                        Layout.fillWidth: true

                        Text {
                            text: "Classification Details"
                            font.pixelSize: Theme.fontBody
                            font.bold: true
                            color: Theme.textSecondary
                        }

                        Item { Layout.fillWidth: true }

                        Text {
                            text: (bridge.defectInferenceTime * 1000).toFixed(1) + " ms"
                            font.pixelSize: Theme.fontSmall
                            color: Theme.textDim
                        }
                    }

                    Repeater {
                        model: {
                            var probs = {}
                            try { probs = JSON.parse(bridge.defectProbsJson || "{}") } catch(e) {}
                            return Object.keys(probs).sort(function(a, b) {
                                return probs[b] - probs[a]
                            })
                        }

                        ConfidenceBar {
                            Layout.fillWidth: true
                            label: "Class " + modelData
                            confidence: {
                                var probs = {}
                                try { probs = JSON.parse(bridge.defectProbsJson || "{}") } catch(e) {}
                                return probs[modelData] || 0
                            }
                            isTop: modelData === bridge.predictedDefectClass
                        }
                    }
                }
            }

            Item { Layout.preferredHeight: 32 }
        }
    }
}
