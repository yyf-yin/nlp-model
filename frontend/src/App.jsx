import React, { useState } from "react";
import {
  Input,
  Button,
  Typography,
  Tag,
  Space,
  message,
  Layout,
  Card,
} from "antd";
import axios from "axios";

const { TextArea } = Input;
const { Title, Paragraph } = Typography;
const { Header, Content, Footer } = Layout;

const App = () => {
  const [text, setText] = useState("");
  const [emotions, setEmotions] = useState([]);
  const [tone, setTone] = useState("");
  const [audience, setAudience] = useState("");
  const [suggestion, setSuggestion] = useState("");

  const handleTextChange = (e) => {
    const newText = e.target.value;
    setText(newText);

    // Auto emotion detection
    axios
      .post("http://localhost:8080/predict", { text: newText })
      .then((res) => {
        setEmotions(res.data.labels);
      })
      .catch((err) => {
        message.error("Emotion detection failed");
      });
  };

  const handleRevise = () => {
    if (!text || !tone || !audience) {
      message.warning("Please fill in all fields");
      return;
    }

    axios
      .post("http://localhost:8080/revise", { text, tone, audience })
      .then((res) => {
        setSuggestion(res.data.suggestion);
      })
      .catch(() => {
        message.error("Failed to revise text");
      });
  };

  return (
    <Layout style={{ minHeight: "100vh" }}>
      <Header style={{ backgroundColor: "#001529" }}>
        <Title style={{ color: "#fff", margin: "10px 0" }} level={3}>
          English Emotion Classifier + Rewriter
        </Title>
      </Header>
      <Content style={{ padding: "40px", maxWidth: 800, margin: "auto" }}>
        <Card bordered>
          <TextArea
            rows={4}
            placeholder="Enter text here"
            value={text}
            onChange={handleTextChange}
          />
          <Space direction="vertical" style={{ marginTop: 20, width: "100%" }}>
            <div>
              <strong>Predicted Emotions:</strong>{" "}
              {emotions.map((emo, index) => (
                <Tag color="blue" key={index}>
                  {emo}
                </Tag>
              ))}
            </div>
            <Input
              placeholder="Desired tone (e.g., polite, confident)"
              value={tone}
              onChange={(e) => setTone(e.target.value)}
            />
            <Input
              placeholder="Audience (e.g., a recruiter, my boss)"
              value={audience}
              onChange={(e) => setAudience(e.target.value)}
            />
            <Button type="primary" onClick={handleRevise}>
              Revise
            </Button>
            {suggestion && (
              <Card
                type="inner"
                title="Revised Suggestion"
                style={{ backgroundColor: "#f6ffed" }}
              >
                <Paragraph>{suggestion}</Paragraph>
              </Card>
            )}
          </Space>
        </Card>
      </Content>
      <Footer style={{ textAlign: "center" }}>
        NLP App ©2025 Created with Ant Design
      </Footer>
    </Layout>
  );
};

export default App;
