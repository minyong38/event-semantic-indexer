// 검색 데모 클라이언트 (Windows, Qt Quick)
//   search_demo.exe [검색 서버] [챗봇 서버]
//   기본 http://localhost:8765 (WSL의 search_server), http://localhost:8770 (WSL의 chat_server)
#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>

int main(int argc, char* argv[]) {
    QGuiApplication app(argc, argv);
    QQmlApplicationEngine engine;
    const QString server = argc > 1 ? QString::fromLocal8Bit(argv[1]) : QStringLiteral("http://localhost:8765");
    const QString chat = argc > 2 ? QString::fromLocal8Bit(argv[2]) : QStringLiteral("http://localhost:8770");
    engine.rootContext()->setContextProperty("serverUrl", server);
    engine.rootContext()->setContextProperty("chatUrl", chat);
    QObject::connect(&engine, &QQmlApplicationEngine::objectCreationFailed, &app,
                     [] { QCoreApplication::exit(-1); }, Qt::QueuedConnection);
    engine.loadFromModule("SearchDemo", "Main");
    return app.exec();
}
