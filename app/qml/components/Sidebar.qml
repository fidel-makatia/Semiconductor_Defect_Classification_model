import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../theme"

Rectangle {
    id: sidebar
    width: Theme.sidebarWidth
    color: Theme.surface

    property int currentIndex: 0
    signal pageSelected(int index)

    readonly property var navItems: [
        { icon: "\u2302", label: "Dashboard" },
        { icon: "\u2316", label: "Defect Detection" },
        { icon: "\u2699", label: "Settings" }
    ]

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 0
        spacing: 0

        // Logo / Brand area
        Item {
            Layout.fillWidth: true
            Layout.preferredHeight: 80

            ColumnLayout {
                anchors.centerIn: parent
                spacing: 2

                Text {
                    text: "SemiAI"
                    font.pixelSize: Theme.fontHeader
                    font.bold: true
                    color: Theme.accent
                    Layout.alignment: Qt.AlignHCenter
                }
                Text {
                    text: "Evaluation Suite"
                    font.pixelSize: Theme.fontTiny
                    color: Theme.textDim
                    Layout.alignment: Qt.AlignHCenter
                    font.letterSpacing: 2
                }
            }
        }

        // Divider
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 1
            Layout.leftMargin: 20
            Layout.rightMargin: 20
            color: Theme.divider
        }

        Item { Layout.preferredHeight: 16 }

        // Nav Items
        Repeater {
            model: navItems

            delegate: Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 48
                Layout.leftMargin: 12
                Layout.rightMargin: 12
                radius: 8
                color: currentIndex === index ? Theme.accent + "20" : hoverArea.containsMouse ? Theme.surfaceLight : "transparent"

                Behavior on color {
                    ColorAnimation { duration: Theme.animFast }
                }

                // Active indicator bar
                Rectangle {
                    width: 3
                    height: parent.height - 16
                    anchors.left: parent.left
                    anchors.verticalCenter: parent.verticalCenter
                    radius: 2
                    color: Theme.accent
                    visible: currentIndex === index

                    Behavior on visible {
                        PropertyAnimation { duration: Theme.animFast }
                    }
                }

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 16
                    anchors.rightMargin: 16
                    spacing: 12

                    Text {
                        text: modelData.icon
                        font.pixelSize: 18
                        color: currentIndex === index ? Theme.accent : Theme.textSecondary
                        Layout.preferredWidth: 24
                        horizontalAlignment: Text.AlignHCenter

                        Behavior on color {
                            ColorAnimation { duration: Theme.animFast }
                        }
                    }

                    Text {
                        text: modelData.label
                        font.pixelSize: Theme.fontBody
                        color: currentIndex === index ? Theme.textPrimary : Theme.textSecondary
                        Layout.fillWidth: true

                        Behavior on color {
                            ColorAnimation { duration: Theme.animFast }
                        }
                    }
                }

                MouseArea {
                    id: hoverArea
                    anchors.fill: parent
                    hoverEnabled: true
                    cursorShape: Qt.PointingHandCursor
                    onClicked: {
                        currentIndex = index
                        sidebar.pageSelected(index)
                    }
                }
            }
        }

        Item { Layout.fillHeight: true }

        // Bottom version info
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 1
            Layout.leftMargin: 20
            Layout.rightMargin: 20
            color: Theme.divider
        }

        Text {
            text: "v1.0.0"
            font.pixelSize: Theme.fontTiny
            color: Theme.textDim
            Layout.alignment: Qt.AlignHCenter
            Layout.bottomMargin: 16
            Layout.topMargin: 12
        }
    }

    // Right border
    Rectangle {
        width: 1
        height: parent.height
        anchors.right: parent.right
        color: Theme.divider
    }
}
