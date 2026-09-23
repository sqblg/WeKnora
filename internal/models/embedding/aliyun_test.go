package embedding

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/Tencent/WeKnora/internal/types"
)

func TestQwenVLModelUsesAliyunMultimodalEndpoint(t *testing.T) {
	t.Setenv("SSRF_WHITELIST", "127.0.0.1")

	var requestPath string
	var requestBody AliyunEmbedRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestPath = r.URL.Path
		if err := json.NewDecoder(r.Body).Decode(&requestBody); err != nil {
			t.Errorf("decode request: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"output":{"embeddings":[{"index":0,"type":"text","embedding":[0.1,0.2]}]}}`))
	}))
	defer server.Close()

	embedder, err := newEmbedder(Config{
		Source:     types.ModelSourceRemote,
		Provider:   "aliyun",
		BaseURL:    server.URL + "/compatible-mode/v1",
		ModelName:  "qwen2.5-vl-embedding",
		APIKey:     "test-key",
		Dimensions: 1024,
	}, nil, nil)
	if err != nil {
		t.Fatalf("newEmbedder: %v", err)
	}
	if _, ok := embedder.(*AliyunEmbedder); !ok {
		t.Fatalf("qwen2.5-vl-embedding routed to %T, want *AliyunEmbedder", embedder)
	}
	got, err := embedder.BatchEmbed(context.Background(), []string{"营业时间"})
	if err != nil {
		t.Fatalf("BatchEmbed: %v", err)
	}
	if requestPath != AliyunMultimodalEmbeddingEndpoint {
		t.Fatalf("request path = %q, want %q", requestPath, AliyunMultimodalEmbeddingEndpoint)
	}
	if requestBody.Model != "qwen2.5-vl-embedding" || len(requestBody.Input.Contents) != 1 || requestBody.Input.Contents[0].Text != "营业时间" {
		t.Fatalf("unexpected multimodal request: %+v", requestBody)
	}
	if !reflect.DeepEqual(got, [][]float32{{0.1, 0.2}}) {
		t.Fatalf("BatchEmbed = %v, want [[0.1 0.2]]", got)
	}
}

func TestAliyunMultimodalBatchUsesResponseIndex(t *testing.T) {
	t.Setenv("SSRF_WHITELIST", "127.0.0.1")

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"output":{"embeddings":[{"index":1,"type":"text","embedding":[0.2]},{"index":0,"type":"text","embedding":[0.1]}]}}`))
	}))
	defer server.Close()

	embedder, err := NewAliyunEmbedder("test-key", server.URL, "qwen2.5-vl-embedding", 0, 1024, "test-model", nil)
	if err != nil {
		t.Fatalf("NewAliyunEmbedder: %v", err)
	}
	got, err := embedder.BatchEmbed(context.Background(), []string{"first", "second"})
	if err != nil {
		t.Fatalf("BatchEmbed: %v", err)
	}
	if !reflect.DeepEqual(got, [][]float32{{0.1}, {0.2}}) {
		t.Fatalf("BatchEmbed = %v, want [[0.1] [0.2]]", got)
	}
}

func TestAliyunMultimodalBatchRetainsLegacyTextIndex(t *testing.T) {
	t.Setenv("SSRF_WHITELIST", "127.0.0.1")

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"output":{"embeddings":[{"text_index":1,"embedding":[0.2]},{"text_index":0,"embedding":[0.1]}]}}`))
	}))
	defer server.Close()

	embedder, err := NewAliyunEmbedder("test-key", server.URL, "tongyi-embedding-vision-plus", 0, 1024, "test-model", nil)
	if err != nil {
		t.Fatalf("NewAliyunEmbedder: %v", err)
	}
	got, err := embedder.BatchEmbed(context.Background(), []string{"first", "second"})
	if err != nil {
		t.Fatalf("BatchEmbed: %v", err)
	}
	if !reflect.DeepEqual(got, [][]float32{{0.1}, {0.2}}) {
		t.Fatalf("BatchEmbed = %v, want [[0.1] [0.2]]", got)
	}
}

func TestAliyunEmbeddingModelRoutingRetainsTextAndQwenVL(t *testing.T) {
	t.Setenv("SSRF_WHITELIST", "127.0.0.1")
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	defer server.Close()

	for _, tt := range []struct {
		name           string
		wantMultimodal bool
	}{
		{name: "qwen2.5-vl-embedding", wantMultimodal: true},
		{name: "qwen3-vl-embedding", wantMultimodal: true},
		{name: "tongyi-embedding-vision-plus", wantMultimodal: true},
		{name: "text-embedding-v4", wantMultimodal: false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			embedder, err := newEmbedder(Config{
				Source: types.ModelSourceRemote, Provider: "aliyun", BaseURL: server.URL + "/compatible-mode/v1", ModelName: tt.name, APIKey: "test-key",
			}, nil, nil)
			if err != nil {
				t.Fatalf("newEmbedder: %v", err)
			}
			_, multimodal := embedder.(*AliyunEmbedder)
			if multimodal != tt.wantMultimodal {
				t.Fatalf("%q routed to %T, wantMultimodal=%t", tt.name, embedder, tt.wantMultimodal)
			}
		})
	}
}
