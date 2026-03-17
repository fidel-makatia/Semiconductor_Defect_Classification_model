import QtQuick
import QtQuick.Layouts
import "../theme"

Item {
    id: bar

    property string label: "Class 0"
    property real confidence: 0.0  // 0.0 to 1.0
    property bool isTop: false
    property color barColor: isTop ? Theme.accent : Theme.surfaceLight

    implicitWidth: 300
    implicitHeight: 32

    RowLayout {
        anchors.fill: parent
        spacing: 12

        // Class label
        Text {
            text: bar.label
            font.pixelSize: Theme.fontSmall
            font.bold: bar.isTop
            color: bar.isTop ? Theme.textPrimary : Theme.textSecondary
            Layout.preferredWidth: 60
            horizontalAlignment: Text.AlignRight
            elide: Text.ElideRight
        }

        // Bar background
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 20
            radius: 4
            color: Theme.surface
            border.color: Theme.divider
            border.width: 1
            clip: true

            // Fill bar
            Rectangle {
                width: parent.width * bar.confidence
                height: parent.height
                radius: 4
                color: bar.barColor
                opacity: bar.isTop ? 0.9 : 0.5

                Behavior on width {
                    NumberAnimation {
                        duration: Theme.animSlow
                        easing.type: Easing.OutCubic
                    }
                }

                // Shimmer effect on top result
                Rectangle {
                    visible: bar.isTop && bar.confidence > 0
                    width: 40
                    height: parent.height
                    radius: 4
                    gradient: Gradient {
                        orientation: Gradient.Horizontal
                        GradientStop { position: 0.0; color: "transparent" }
                        GradientStop { position: 0.5; color: Qt.rgba(1, 1, 1, 0.1) }
                        GradientStop { position: 1.0; color: "transparent" }
                    }

                    SequentialAnimation on x {
                        running: bar.isTop && bar.confidence > 0
                        loops: Animation.Infinite
                        NumberAnimation {
                            from: -40
                            to: bar.confidence * bar.width
                            duration: 2000
                            easing.type: Easing.InOutQuad
                        }
                        PauseAnimation { duration: 1000 }
                    }
                }
            }
        }

        // Percentage text
        Text {
            text: (bar.confidence * 100).toFixed(1) + "%"
            font.pixelSize: Theme.fontSmall
            font.bold: bar.isTop
            color: bar.isTop ? Theme.accent : Theme.textDim
            Layout.preferredWidth: 50
            horizontalAlignment: Text.AlignRight

            Behavior on color {
                ColorAnimation { duration: Theme.animNormal }
            }
        }
    }
}
