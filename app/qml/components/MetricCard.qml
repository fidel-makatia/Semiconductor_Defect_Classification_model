import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../theme"

Rectangle {
    id: card

    property string title: "Metric"
    property string value: "—"
    property string subtitle: ""
    property string iconText: ""
    property color accentColor: Theme.accent
    property bool loading: false

    implicitWidth: 220
    implicitHeight: 120
    radius: Theme.cardRadius
    color: Theme.surface
    border.color: Theme.divider
    border.width: 1

    // Subtle glow on hover
    Rectangle {
        anchors.fill: parent
        radius: parent.radius
        color: "transparent"
        border.color: card.accentColor
        border.width: hoverArea.containsMouse ? 1 : 0
        opacity: 0.3

        Behavior on border.width {
            NumberAnimation { duration: Theme.animFast }
        }
    }

    MouseArea {
        id: hoverArea
        anchors.fill: parent
        hoverEnabled: true
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 16
        spacing: 6

        RowLayout {
            Layout.fillWidth: true
            spacing: 10

            // Icon circle
            Rectangle {
                width: 36
                height: 36
                radius: 18
                color: card.accentColor + "20"

                Text {
                    anchors.centerIn: parent
                    text: card.iconText
                    font.pixelSize: 16
                    color: card.accentColor
                }
            }

            Text {
                text: card.title
                font.pixelSize: Theme.fontSmall
                color: Theme.textSecondary
                Layout.fillWidth: true
                elide: Text.ElideRight
            }
        }

        Item { Layout.fillHeight: true }

        // Value
        Text {
            text: card.loading ? "Loading..." : card.value
            font.pixelSize: card.value.length > 16 ? Theme.fontBody : Theme.fontHeader
            font.bold: true
            color: card.loading ? Theme.textDim : Theme.textPrimary
            Layout.fillWidth: true
            elide: Text.ElideRight
            maximumLineCount: 1

            Behavior on color {
                ColorAnimation { duration: Theme.animNormal }
            }

            // Loading shimmer
            SequentialAnimation on opacity {
                running: card.loading
                loops: Animation.Infinite
                NumberAnimation { to: 0.4; duration: 600 }
                NumberAnimation { to: 1.0; duration: 600 }
            }
        }

        // Subtitle
        Text {
            text: card.subtitle
            font.pixelSize: Theme.fontTiny
            color: Theme.textDim
            visible: card.subtitle !== ""
            Layout.fillWidth: true
            elide: Text.ElideRight
        }
    }
}
