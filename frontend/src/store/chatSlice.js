import { createSlice, createAsyncThunk, nanoid } from "@reduxjs/toolkit";
import { api } from "../api/api";

export const sendMessage = createAsyncThunk(
  "chat/sendMessage",
  async ({ message, hcpId }, { getState }) => {
    const state = getState().chat;
    const interactionsState = getState().interactions;
    const response = await api.sendChatMessage({
      session_id: state.sessionId,
      hcp_id: hcpId || null,
      message,
      history: state.messages.map((m) => ({ role: m.role, content: m.content })),
      draft_interaction: interactionsState.draftInteraction || null,
    });
    return response;
  }
);

const chatSlice = createSlice({
  name: "chat",
  initialState: {
    sessionId: nanoid(),
    messages: [], // { id, role, content, toolCalls? }
    status: "idle",
  },
  reducers: {
    addUserMessage(state, action) {
      state.messages.push({ id: nanoid(), role: "user", content: action.payload });
    },
    resetChat(state) {
      state.sessionId = nanoid();
      state.messages = [];
      state.status = "idle";
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(sendMessage.pending, (state) => {
        state.status = "loading";
      })
      .addCase(sendMessage.fulfilled, (state, action) => {
        state.status = "idle";
        state.messages.push({
          id: nanoid(),
          role: "assistant",
          content: action.payload.reply,
          toolCalls: action.payload.tool_calls || [],
        });
      })
      .addCase(sendMessage.rejected, (state, action) => {
        state.status = "idle";
        state.messages.push({
          id: nanoid(),
          role: "assistant",
          content: "I had a momentary hiccup — please send your message again and I'll pick up right where we left off.",
        });
      });
  },
});

export const { addUserMessage, resetChat } = chatSlice.actions;
export default chatSlice.reducer;
