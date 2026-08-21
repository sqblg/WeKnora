package container

import (
	"bytes"
	"context"
	"log"
	"strings"
	"testing"

	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
)

func TestGormLoggerNeverWritesBoundParameterValues(t *testing.T) {
	var output bytes.Buffer
	secret := "wk-public-beta-secret-must-not-reach-logs"
	logger := newGormLogger(log.New(&output, "", 0), "warn")

	db, err := gorm.Open(sqlite.Open(":memory:"), &gorm.Config{Logger: logger})
	if err != nil {
		t.Fatal(err)
	}
	if result := db.Exec("INSERT INTO missing_auth_tokens (token) VALUES (?)", secret); result.Error == nil {
		t.Fatal("expected the missing table to force a parameterized SQL error log")
	}

	if strings.Contains(output.String(), secret) {
		t.Fatalf("GORM log exposed a bound secret: %s", output.String())
	}
}

func TestGormLoggerRejectsVerboseProductionLevels(t *testing.T) {
	var output bytes.Buffer
	logger := newGormLogger(log.New(&output, "", 0), "info")

	logger.Info(context.Background(), "sensitive info event")

	if output.Len() != 0 {
		t.Fatalf("production GORM logger accepted info output: %s", output.String())
	}
}
