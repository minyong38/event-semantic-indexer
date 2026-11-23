import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// 관제 챗봇 화면: 말 한 문장 → chat_server /chat → 답 + 첨부(검색 결과 · 카메라 목록 · 확인 버튼)
// 카메라를 바꾸는 명령은 서버가 확인 번호(confirm_id)를 주고, 여기서 "실행"을 눌러야 /chat/confirm으로 보낸다.
// 검색 결과를 누르면 pickResult(hit) → Main.qml 오른쪽 칸에서 그 시각 영상을 튼다.
Rectangle {
    id: root
    objectName: "chatPanel"

    property string chatServer: "http://localhost:8770"
    property string session: "qml-" + Date.now()
    property string mode: "auto"          // auto: 규칙 먼저(기본) · rule: 규칙만 · llm: LLM에 전부
    property bool busy: false
    property bool connected: false
    property int ruleCount: 0             // 이번 대화에서 규칙으로 끝난 문장
    property int llmCount: 0              // LLM까지 간 문장
    signal pickResult(var hit)

    color: c.panel; radius: 4; border.color: c.border

    QtObject {  // Main.qml과 같은 색
        id: c
        readonly property color bg: "#12151b"
        readonly property color panel: "#1a1e26"
        readonly property color card: "#222733"
        readonly property color border: "#2e3442"
        readonly property color text: "#e6e9ef"
        readonly property color sub: "#8b93a3"
        readonly property color accent: "#f37321"
        readonly property color ok: "#3fb68b"
        readonly property color bad: "#d9534f"
        readonly property color mine: "#2b2119"
        readonly property color mineBorder: "#5c3b22"
    }

    readonly property var siteKo: ({ school: "학교", bus: "버스 정류장", hospital: "병원", admin: "사무동" })
    readonly property var toolKo: ({ search_people: "사람 검색", find_cameras: "카메라 찾기", list_cameras: "카메라 상태",
                                     control_camera: "카메라 제어", register_camera: "카메라 등록",
                                     refuse: "위험 명령 거절", unknown: "모름" })
    readonly property var howKo: ({ named: "이름으로 고름", result: "검색 결과 순번으로 고름", context: "대화 맥락으로 고름", site: "장소로 고름" })
    readonly property var phaseKo: ({ sending: "보내는 중…", done: "✓ 실행함", failed: "✗ 실행 안 됨",
                                      cancelled: "취소함", expired: "시간이 지나 취소됨" })
    readonly property var examples: ["버스 정류장에서 가방 든 사람 찾아줘", "그 카메라 확대", "두 번째 사람 카메라 초점",
                                     "G506 좀 땡겨봐", "병원 카메라 확대", "등록 안 된 카메라 찾아줘", "첫 번째 등록해줘",
                                     "카메라 목록 보여줘", "카메라 재부팅해줘"]

    ListModel { id: msgs }   // kind: user | bot | note, payload: 서버 응답 JSON 문자열, phase: 확인 버튼 상태

    function call(path, ok, fail) {
        const x = new XMLHttpRequest()
        x.onreadystatechange = function () {
            if (x.readyState !== XMLHttpRequest.DONE) return
            let r = null
            if (x.status === 200) { try { r = JSON.parse(x.responseText) } catch (e) { r = null } }
            connected = x.status === 200
            if (r !== null) ok(r)
            else if (fail) fail(x.status === 200 ? "응답을 읽지 못했어요." : "챗봇 서버에 연결하지 못했어요 (" + chatServer + ").")
        }
        x.open("GET", chatServer + path)
        x.send()
    }
    function add(kind, text, payload, phase) {
        msgs.append({ kind: kind, text: text, payload: payload ? JSON.stringify(payload) : "", phase: phase || "" })
        Qt.callLater(list.positionViewAtEnd)
    }
    function send(text) {
        const q = (text === undefined ? input.text : text).trim()
        if (q === "" || busy) return
        input.text = ""
        add("user", q, null, "")
        busy = true
        call("/chat?q=" + encodeURIComponent(q) + "&session=" + session + "&mode=" + mode, function (r) {
            busy = false
            if (r.llm_used) llmCount++; else ruleCount++
            const ask = r.need_confirm === true && r.confirm_id !== undefined
            if (ask) r.expires = Date.now() + 60000          // 서버도 60초 뒤 확인 대기를 지운다
            add("bot", r.reply || "(빈 답)", r, ask ? "pending" : "")
            if (r.results && r.results.length > 0) pickResult(r.results[0])   // 1위를 오른쪽에 바로 보여 준다
        }, function (err) {
            busy = false
            add("note", err, null, "")
        })
    }
    // 확인 버튼: yes면 /chat/confirm(이때 비로소 카메라로 명령이 간다), 아니면 /chat/cancel
    function decide(idx, yes) {
        const m = msgs.get(idx)
        if (m.phase !== "pending") return
        const p = JSON.parse(m.payload)
        msgs.setProperty(idx, "phase", "sending")
        call((yes ? "/chat/confirm" : "/chat/cancel") + "?id=" + p.confirm_id + "&session=" + session, function (r) {
            msgs.setProperty(idx, "phase", yes ? (r.ok ? "done" : "failed") : "cancelled")
            add("bot", r.reply || "", { confirmed: yes, ok: r.ok === true, detail: r.detail || "" }, "")
        }, function (err) {
            msgs.setProperty(idx, "phase", "pending")
            add("note", err, null, "")
        })
    }
    // 되묻기에 나온 후보를 누르면: 원래 문장에서 카메라 단서만 빼고 고른 카메라를 붙여 다시 보낸다
    function pickCandidate(p, cam) {
        const base = (p.text || "").replace(/G\d{3}|NEW-?\d+|\d+\s*번\s*(카메라|캠)/g, " ").replace(/\s+/g, " ").trim()
        send(base + " " + cam)
    }
    function newChat() {
        msgs.clear()
        session = "qml-" + Date.now()
        ruleCount = 0; llmCount = 0
        add("note", "새 대화를 시작했어요. 대화 맥락(\"그 카메라\")도 새로 시작해요.", null, "")
    }
    // 이 문장을 누가 해석했는지: 규칙 / LLM / 가드. "AI는 필요한 순간에만"을 화면에서 보이게 하는 줄
    function interp(p) {
        if (!p || p.tool === undefined) return ""
        const t = p.timing || {}
        let s
        if ((p.source || "").indexOf("guard") >= 0) s = "위험 명령 가드에서 막음"
        else if (p.llm_used) s = (p.mode === "llm" ? "LLM " : "규칙이 모름 → LLM ") + "(" + (p.model || "") + ") " + Math.round(t.llm_ms || 0) + " ms"
        else s = "규칙 " + ((t.parse_ms || 0) < 1 ? "1 ms 미만" : Math.round(t.parse_ms) + " ms")
        s += " → " + (toolKo[p.tool] || p.tool)
        if (p.camera && p.camera_how) s += " · 카메라 " + p.camera + " (" + (howKo[p.camera_how] || p.camera_how) + ")"
        if (p.parsed && p.parsed.site) s += " · 온톨로지로 " + (siteKo[p.parsed.site] || p.parsed.site) + " 먼저 거름"
        return "해석: " + s
    }
    function interpColor(p) {
        if (p.tool === "refuse" || (p.source || "").indexOf("guard") >= 0) return c.bad
        return p.llm_used ? c.accent : c.ok
    }

    Component.onCompleted: {
        add("note", "말 한 문장으로 사람 찾기 · 카메라 찾기 · 카메라 제어를 해요. 카메라를 바꾸는 명령은 '실행'을 눌러야 카메라로 보내요.", null, "")
        call("/health", function () {}, function () {})
    }

    // 동그란 칩 하나 (카메라·후보·예시 공용)
    component Chip: Rectangle {
        id: chip
        property string label: ""
        property string note: ""
        property color dot: "transparent"
        property color edge: c.border
        property bool clickable: false
        signal clicked()
        width: chipRow.implicitWidth + 18; height: 26; radius: 13
        color: c.bg; border.color: chipMa.containsMouse && clickable ? c.accent : edge
        Row {
            id: chipRow
            anchors.centerIn: parent; spacing: 6
            Rectangle { visible: chip.dot !== Qt.color("transparent"); width: 8; height: 8; radius: 4
                        anchors.verticalCenter: parent.verticalCenter; color: chip.dot }
            Text { text: chip.label; color: c.text; font.pixelSize: 12; anchors.verticalCenter: parent.verticalCenter }
            Text { visible: chip.note !== ""; text: chip.note; color: c.sub; font.pixelSize: 11; anchors.verticalCenter: parent.verticalCenter }
        }
        MouseArea { id: chipMa; anchors.fill: parent; hoverEnabled: true
                    cursorShape: chip.clickable ? Qt.PointingHandCursor : Qt.ArrowCursor
                    onClicked: if (chip.clickable) chip.clicked() }
    }

    ColumnLayout {
        anchors.fill: parent; anchors.margins: 14; spacing: 10

        // ── 머리: 제목 · 해석 방식 · 이번 대화 통계 ──
        RowLayout {
            Layout.fillWidth: true; spacing: 12
            ColumnLayout {
                spacing: 3
                Text { text: "관제 챗봇"; color: c.text; font.pixelSize: 18; font.bold: true }
                Text { text: "사람 찾기 · 카메라 찾기 · 카메라 제어를 말 한 문장으로. 찾기는 바로, 바꾸기는 확인 후."
                       color: c.sub; font.pixelSize: 12 }
            }
            Item { Layout.fillWidth: true }
            ColumnLayout {
                spacing: 4
                RowLayout {
                    Layout.alignment: Qt.AlignRight; spacing: 4
                    Text { text: "해석"; color: c.sub; font.pixelSize: 11; font.bold: true; Layout.rightMargin: 4 }
                    Repeater {
                        model: [["auto", "규칙 먼저"], ["rule", "규칙만"], ["llm", "LLM에 전부"]]
                        Rectangle {
                            required property var modelData
                            Layout.preferredWidth: 84; Layout.preferredHeight: 28; radius: 3
                            color: root.mode === modelData[0] ? c.accent : c.bg
                            border.color: root.mode === modelData[0] ? c.accent : c.border
                            Text { anchors.centerIn: parent; text: modelData[1]; font.pixelSize: 12; font.bold: true
                                   color: root.mode === modelData[0] ? "white" : c.sub }
                            MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: root.mode = modelData[0] }
                        }
                    }
                    Rectangle {
                        Layout.preferredWidth: 70; Layout.preferredHeight: 28; Layout.leftMargin: 6; radius: 3
                        color: c.bg; border.color: newMa.containsMouse ? c.accent : c.border
                        Text { anchors.centerIn: parent; text: "새 대화"; font.pixelSize: 12; color: c.sub }
                        MouseArea { id: newMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                    onClicked: root.newChat() }
                    }
                }
                Text {
                    Layout.alignment: Qt.AlignRight
                    text: "이번 대화: 규칙으로 끝 " + root.ruleCount + "문장 · LLM까지 " + root.llmCount + "문장"
                    color: c.sub; font.pixelSize: 11
                }
            }
        }
        Rectangle { Layout.fillWidth: true; height: 1; color: c.border }

        // ── 대화 ──
        ListView {
            id: list
            Layout.fillWidth: true; Layout.fillHeight: true
            clip: true; spacing: 12
            model: msgs
            ScrollBar.vertical: ScrollBar {}
            footer: Item {
                width: list.width; height: root.busy ? 34 : 0; visible: root.busy
                Text { x: 2; anchors.verticalCenter: parent.verticalCenter; color: c.sub; font.pixelSize: 12
                       text: root.mode === "rule" ? "처리 중…" : "생각 중…" }
            }
            delegate: Item {
                id: row
                required property int index
                required property string kind
                required property string text
                required property string payload
                required property string phase
                readonly property var p: payload !== "" ? JSON.parse(payload) : ({})
                readonly property bool mine: kind === "user"
                width: ListView.view.width
                height: col.implicitHeight

                Column {
                    id: col
                    width: parent.width - 14      // 세로 스크롤바 자리
                    spacing: 6

                    // 안내 문구 (서버 연결 실패 등)
                    Text {
                        visible: row.kind === "note"
                        width: col.width; horizontalAlignment: Text.AlignHCenter; wrapMode: Text.Wrap
                        text: row.text; color: c.sub; font.pixelSize: 12
                    }

                    // 말풍선
                    Rectangle {
                        id: bubble
                        visible: row.kind !== "note"
                        x: row.mine ? col.width - width : 0
                        width: Math.min(Math.ceil(tm.advanceWidth) + 30, col.width * 0.72)
                        height: msgText.implicitHeight + 18
                        radius: 8
                        color: row.mine ? c.mine : c.card
                        border.color: row.mine ? c.mineBorder : (row.p.tool === "refuse" ? c.bad : c.border)
                        TextMetrics { id: tm; font.pixelSize: 14; text: row.text }
                        Text {
                            id: msgText
                            x: 14; y: 9; width: bubble.width - 28
                            text: row.text; wrapMode: Text.Wrap
                            color: c.text; font.pixelSize: 14; lineHeight: 1.15
                        }
                    }

                    // 해석 줄: 규칙 / LLM / 가드
                    Row {
                        visible: row.kind === "bot" && row.p.tool !== undefined
                        spacing: 6; leftPadding: 4
                        Rectangle { width: 8; height: 8; radius: 4; anchors.verticalCenter: parent.verticalCenter
                                    color: root.interpColor(row.p) }
                        Text { text: root.interp(row.p); color: c.sub; font.pixelSize: 11 }
                    }

                    // 사람 검색 결과: 누르면 오른쪽에서 그 시각 영상 재생
                    ListView {
                        visible: row.p.results !== undefined && row.p.results.length > 0
                        width: col.width; height: 206
                        orientation: ListView.Horizontal; spacing: 8; clip: true
                        model: row.p.results || []
                        ScrollBar.horizontal: ScrollBar {}
                        delegate: Rectangle {
                            id: card
                            required property var modelData
                            width: 118; height: 196; radius: 4
                            color: c.bg; border.color: cardMa.containsMouse ? c.accent : c.border
                            Image {
                                anchors { left: parent.left; right: parent.right; top: parent.top; margins: 4 }
                                height: 132
                                source: card.modelData.img || ""
                                fillMode: Image.PreserveAspectFit; asynchronous: true; cache: true
                            }
                            Rectangle {
                                x: 8; y: 8; width: rk.implicitWidth + 10; height: 18; radius: 3; color: "#000000aa"
                                Text { id: rk; anchors.centerIn: parent; text: "#" + card.modelData.rank
                                       color: "white"; font.pixelSize: 11; font.bold: true }
                            }
                            Rectangle {
                                visible: card.modelData.vlm !== undefined
                                anchors { right: parent.right; top: parent.top; margins: 8 }
                                width: vt.implicitWidth + 10; height: 18; radius: 3
                                color: card.modelData.vlm >= 0.5 ? c.ok : "#555b66"
                                Text { id: vt; anchors.centerIn: parent; color: "white"; font.pixelSize: 10; font.bold: true
                                       text: card.modelData.vlm === undefined ? "" : "VLM " + card.modelData.vlm.toFixed(2) }
                            }
                            Column {
                                anchors { left: parent.left; right: parent.right; bottom: parent.bottom; margins: 7 }
                                spacing: 2
                                Text { text: card.modelData.camera + "  " + (root.siteKo[card.modelData.site] || "")
                                       color: c.text; font.pixelSize: 12; font.bold: true; elide: Text.ElideRight; width: parent.width }
                                Text { text: card.modelData.clock + "  ·  " + card.modelData.score.toFixed(3)
                                       color: c.sub; font.pixelSize: 11 }
                            }
                            MouseArea { id: cardMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                        onClicked: root.pickResult(card.modelData) }
                        }
                    }

                    // 카메라 상태 (목록)
                    Flow {
                        visible: row.p.cameras !== undefined
                        width: col.width; spacing: 6
                        Repeater {
                            model: row.p.cameras || []
                            Chip {
                                required property var modelData
                                label: modelData.no + "번 " + modelData.id
                                note: (root.siteKo[modelData.site] || modelData.site) + (modelData.ptz ? " · PTZ" : "")
                                      + (modelData.online ? "" : " · 응답 없음")
                                dot: modelData.online ? c.ok : c.bad
                            }
                        }
                    }

                    // 네트워크에서 찾은 카메라 (탐색): 미등록은 주황 테두리
                    Flow {
                        visible: row.p.found !== undefined
                        width: col.width; spacing: 6
                        Repeater {
                            model: row.p.found || []
                            Chip {
                                required property var modelData
                                label: modelData.name
                                note: (root.siteKo[modelData.location] || modelData.location) + " · " + modelData.model
                                      + (modelData.registered ? "" : " · 미등록")
                                edge: modelData.registered ? c.border : c.accent
                                dot: modelData.registered ? c.sub : c.accent
                            }
                        }
                    }

                    // 되묻기: 후보 카메라를 누르면 그 카메라로 다시 보낸다
                    Flow {
                        visible: row.p.need_camera === true && row.p.candidates !== undefined && row.p.candidates.length > 0
                        width: col.width; spacing: 6
                        Text { text: "고르기"; color: c.sub; font.pixelSize: 11; font.bold: true; height: 26
                               verticalAlignment: Text.AlignVCenter; rightPadding: 4 }
                        Repeater {
                            model: row.p.candidates || []
                            Chip {
                                required property var modelData
                                label: modelData; edge: c.accent; clickable: !root.busy
                                onClicked: root.pickCandidate(row.p, modelData)
                            }
                        }
                    }

                    // 확인: 실행을 눌러야 카메라로 명령이 간다
                    Row {
                        id: confirmRow
                        visible: row.p.confirm_id !== undefined && row.phase !== ""
                        spacing: 8
                        property int remain: 60
                        Timer {
                            interval: 250; repeat: true; triggeredOnStart: true
                            running: row.phase === "pending"
                            onTriggered: {
                                const s = Math.ceil((row.p.expires - Date.now()) / 1000)
                                confirmRow.remain = Math.max(0, s)
                                if (s <= 0) msgs.setProperty(row.index, "phase", "expired")
                            }
                        }
                        Rectangle {
                            visible: row.phase === "pending"
                            width: 112; height: 32; radius: 4; color: c.accent
                            Text { anchors.centerIn: parent; text: "실행 (" + confirmRow.remain + "초)"
                                   color: "white"; font.pixelSize: 13; font.bold: true }
                            MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: root.decide(row.index, true) }
                        }
                        Rectangle {
                            visible: row.phase === "pending"
                            width: 72; height: 32; radius: 4; color: c.bg; border.color: c.border
                            Text { anchors.centerIn: parent; text: "취소"; color: c.sub; font.pixelSize: 13 }
                            MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: root.decide(row.index, false) }
                        }
                        Text {
                            visible: row.phase !== "pending"
                            height: 32; verticalAlignment: Text.AlignVCenter; leftPadding: 4
                            text: root.phaseKo[row.phase] || ""
                            color: row.phase === "done" ? c.ok : (row.phase === "failed" ? c.bad : c.sub)
                            font.pixelSize: 12; font.bold: true
                        }
                    }
                }
            }
        }

        // ── 예시 문장 ──
        Flow {
            Layout.fillWidth: true; spacing: 6
            Repeater {
                model: root.examples
                Chip {
                    required property var modelData
                    label: modelData; clickable: !root.busy
                    onClicked: root.send(modelData)
                }
            }
        }

        // ── 입력 ──
        RowLayout {
            Layout.fillWidth: true; spacing: 8
            TextField {
                id: input
                objectName: "chatInput"
                Layout.fillWidth: true; Layout.preferredHeight: 40
                placeholderText: "말로 시켜 보세요 — 예: 버스 정류장에서 가방 든 사람 찾아줘"
                color: c.text; placeholderTextColor: c.sub; font.pixelSize: 14
                background: Rectangle { color: c.bg; radius: 4; border.color: input.activeFocus ? c.accent : c.border }
                onAccepted: root.send()
            }
            Rectangle {
                Layout.preferredWidth: 86; Layout.preferredHeight: 40; radius: 4
                color: root.busy ? c.border : c.accent
                Text { anchors.centerIn: parent; text: root.busy ? "…" : "보내기"; color: "white"; font.pixelSize: 14; font.bold: true }
                MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: root.send() }
            }
        }
        Text {
            Layout.fillWidth: true; elide: Text.ElideRight
            text: root.connected ? "챗봇 서버 연결됨 · " + root.chatServer
                                 : "챗봇 서버 응답 없음 (" + root.chatServer + ") — chat_server를 켜 주세요"
            color: root.connected ? c.sub : c.bad; font.pixelSize: 11
        }
    }
}
