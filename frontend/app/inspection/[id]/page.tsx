import { Workspace } from "@/components/workspace";

export default async function InspectionPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <Workspace view="inspection" inspectionId={id} />;
}